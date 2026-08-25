"""Bus-plan fanout layer assignment (#622): choose each escaping net's layer
so the planned bus trunks can be drawn WITHOUT walling in any other tooth.

Motivating failure (measured, fb_lc_base K=2): SDQ10+SDQ8 routed as facing
F.Cu trunks 0.39mm apart -- with SDQ11's F.Cu tooth between them (0.29mm of
free width where a 0.1 track needs 0.30 with clearance). The trunks connected
AROUND a live same-layer tooth: a bad plan by construction, invisible to any
per-route greedy/rebalance decision because no single route is at fault.

Three constraints, solved jointly at probe-plan time (the only stage that
knows the bus groups before any stub exists):

1. END-MATCHING: a net keeps ONE layer for both combs. The first fanout
   commits; the second reads the board's committed facing teeth as fixed
   variables (`committed_tip_layer`). A net the escape engine fails and
   rescues (dive = bottom layer) self-heals: its committed copper is the
   truth the next fanout matches, not this plan.
2. GROUP NON-INTERLEAVING (the #622 topology law): two groups whose comb
   spans overlap on the same layer force a crossing (Jordan curve) -- they
   must take different layers. Spans are checked in BOTH orderings the plan
   can see: the local comb order (lateral tooth order on the escape side)
   and the far-end order (the probe paths' far endpoints -- the facing
   comb's order, known before the facing fanout exists).
3. NON-BLOCKING (the user's law): no foreign tooth may lie strictly inside
   a group's span on that group's layer -- it could never escape past the
   trunks. Any such tooth is forced to the complement layer.

The solver is interval coloring with groups as atoms: units = (group, side)
plus singleton units for planned non-group nets; must-differ edges from span
overlap in either ordering; K = len(layers) colors; fixed colors from
committed far-end copper and from diff-pair nets (the channel engine forbids
pairs on the top layer, and the pair machinery owns their geometry -- the
plan records, never moves them). A net whose fixed layer disagrees with its
unit's color is EJECTED to its fixed layer (committed copper outranks the
plan) and reported. Infeasibility (3+ mutually interleaved groups on 2
layers) never fails the fanout: the coloring minimizes conflicted edges and
reports each one.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from bga_fanout.probe_plan import FanoutProbePlan, _lateral_key


@dataclass
class FanoutLayerPlan:
    # net_id -> planned layer name (every planned net appears)
    net_layer: Dict[int, str] = field(default_factory=dict)
    # subset whose layer was FIXED (committed facing copper / diff pair)
    fixed: Dict[int, str] = field(default_factory=dict)
    # nets of conflicted, un-committed units -- candidates for the caller's
    # wall-aware side re-allocation retry (#622 lever 2; the human sends
    # exactly these nets out other faces)
    reside_candidates: Set[int] = field(default_factory=set)
    report: dict = field(default_factory=dict)

    def layer_of(self, net_id: int) -> Optional[str]:
        return self.net_layer.get(net_id)


def committed_tip_layer(pcb_data, net_id: int,
                        near_pt: Tuple[float, float]) -> Optional[str]:
    """Layer of the net's committed free tooth tip nearest `near_pt` (our
    ball) -- the attach layer a trunk to this net must use. None when the
    net has no committed segments yet (first-fanout case)."""
    segs = [s for s in pcb_data.segments if s.net_id == net_id]
    if not segs:
        return None
    from collections import Counter
    cnt: Counter = Counter()
    for s in segs:
        cnt[(round(s.start_x, 3), round(s.start_y, 3))] += 1
        cnt[(round(s.end_x, 3), round(s.end_y, 3))] += 1
    tips = []
    for s in segs:
        for (x, y) in ((s.start_x, s.start_y), (s.end_x, s.end_y)):
            if cnt[(round(x, 3), round(y, 3))] == 1:
                tips.append((x, y, s.layer))
    if not tips:
        return None
    x0, y0 = near_pt
    tips.sort(key=lambda t: (t[0] - x0) ** 2 + (t[1] - y0) ** 2)
    return tips[0][2]


def _far_lateral(pts: List[Tuple[float, float]], side: str) -> float:
    """Far-end order key: the probe path's far endpoint's lateral coordinate
    (same lateral axis as the local side, so both orderings are comparable
    along the comb)."""
    lat = 1 if side in ('left', 'right') else 0
    return pts[-1][lat]


def _spans_overlap(a: Tuple[float, float], b: Tuple[float, float],
                   eps: float = 1e-6) -> bool:
    return a[0] < b[1] - eps and b[0] < a[1] - eps


def build_fanout_layer_plan(plan: FanoutProbePlan, pcb_data, grid,
                            layers: List[str],
                            alloc: Dict[Tuple[float, float], str],
                            pair_net_ids: Set[int],
                            verbose: bool = True) -> Optional[FanoutLayerPlan]:
    """Solve the layer assignment for every probed net. Returns None when
    fewer than 2 layers are available (nothing to choose)."""
    if len(layers) < 2:
        return None

    # --- per-net side + order keys -------------------------------------
    side_of: Dict[int, str] = {}
    local_key: Dict[int, float] = {}
    far_key: Dict[int, float] = {}
    for nid, (bx, by) in plan.ball_pos.items():
        k = (round(bx, 3), round(by, 3))
        side = alloc.get(k) or plan.pad_dirs.get(k)
        pts = plan.paths_mm.get(nid)
        if side is None or not pts:
            continue
        lk = _lateral_key(pts, side, grid)
        if lk is None:
            continue
        side_of[nid] = side
        local_key[nid] = lk
        far_key[nid] = _far_lateral(pts, side)
    if not side_of:
        return None

    # --- fixed layers ---------------------------------------------------
    lp = FanoutLayerPlan()
    assumed_pairs: Set[int] = set()
    for nid in side_of:
        cl = committed_tip_layer(pcb_data, nid, plan.ball_pos[nid])
        if cl is not None and cl in layers:
            lp.fixed[nid] = cl
        elif nid in pair_net_ids:
            # The pair escape machinery owns diff-pair geometry (half-edge
            # pairs lawfully take the top layer). Assume the engine's
            # non-top default for the CONSTRAINT solve only -- assumed
            # pairs are dropped from the emitted net_layer below, so the
            # engines never get seeded for them.
            lp.fixed[nid] = layers[1]
            assumed_pairs.add(nid)

    # --- units: (group, side) atoms + singletons ------------------------
    grouped: Set[int] = set()
    units: List[Dict] = []  # {'nets': [...], 'side': s, 'group': bool}
    for gi, group in enumerate(plan.groups):
        by_side: Dict[str, List[int]] = {}
        for nid in group:
            if nid in side_of:
                by_side.setdefault(side_of[nid], []).append(nid)
        for side, nets in by_side.items():
            grouped.update(nets)
            units.append({'nets': nets, 'side': side, 'group': len(nets) > 1,
                          'gi': gi})
    for nid in side_of:
        if nid not in grouped:
            units.append({'nets': [nid], 'side': side_of[nid], 'group': False,
                          'gi': None})

    def _span(u, key):
        vals = [key[n] for n in u['nets'] if n in key]
        return (min(vals), max(vals)) if vals else None

    for u in units:
        u['span_local'] = _span(u, local_key)
        u['span_far'] = _span(u, far_key)

    # --- must-differ edges ----------------------------------------------
    # Group-vs-group: overlapping spans in EITHER ordering (interleaving =
    # forced crossing; containment = walled teeth). Single-vs-group: the
    # single's point key inside the group's span (walled tooth). Singles
    # never constrain each other (no planned trunk between them).
    n_u = len(units)
    edges: Set[Tuple[int, int]] = set()
    for i in range(n_u):
        for j in range(i + 1, n_u):
            a, b = units[i], units[j]
            if a['side'] != b['side']:
                continue
            if not a['group'] and not b['group']:
                continue
            # LOCAL comb spans only. Far-end spans are deliberately NOT
            # edges: with a single facing destination every far span
            # overlaps every other (measured: 8 unsatisfiable conflicts on
            # allwinner DU1), and the far comb's own non-blocking law is
            # enforced when ITS fanout runs -- end-matching hands it this
            # solve's layers as fixed variables, and its solve reports any
            # walls those force. Far spans stay in the report.
            sa, sb = a['span_local'], b['span_local']
            if sa and sb and _spans_overlap(sa, sb):
                edges.add((i, j))

    adj: Dict[int, Set[int]] = {i: set() for i in range(n_u)}
    for i, j in edges:
        adj[i].add(j)
        adj[j].add(i)

    # --- coloring ---------------------------------------------------------
    # Fixed-constrained units first (most fixed members first), then larger
    # units, so the bus sheaves claim the top layer before the singles that
    # must dodge them. Preference order: fixed-majority layer, then the top
    # layer (surface trunks -- the human default), then list order.
    def _fixed_majority(u):
        # Only COMMITTED copper votes -- an assumed pair layer is a
        # constraint placeholder, not a commitment, and letting it vote
        # dragged whole F.Cu-viable groups onto the bottom layer (measured:
        # the SDQS0 pair's assumption pulled a 17-member group to B.Cu).
        votes: Dict[str, int] = {}
        for n in u['nets']:
            f = lp.fixed.get(n)
            if f and n not in assumed_pairs:
                votes[f] = votes.get(f, 0) + 1
        return max(votes, key=votes.get) if votes else None

    order = sorted(range(n_u), key=lambda i: (
        -sum(1 for n in units[i]['nets']
             if n in lp.fixed and n not in assumed_pairs),
        -len(units[i]['nets'])))
    color: Dict[int, str] = {}
    violations: List[str] = []
    for i in order:
        u = units[i]
        fm = _fixed_majority(u)
        prefs = ([fm] if fm else []) + [l for l in layers if l != fm]
        best, best_conf = None, None
        for lay in prefs:
            conf = sum(1 for j in adj[i] if color.get(j) == lay)
            if best_conf is None or conf < best_conf:
                best, best_conf = lay, conf
        color[i] = best
        if best_conf:
            names = [plan.net_names.get(n, str(n)) for n in u['nets']]
            _nbrs = ['+'.join(plan.net_names.get(n, str(n)).split('/')[-1]
                              for n in units[j]['nets'][:3])
                     for j in adj[i] if color.get(j) == best]
            violations.append(
                f"{'+'.join(names[:4])}{'...' if len(names) > 4 else ''} "
                f"on {best} conflicts with [{'; '.join(_nbrs)}] "
                f"(side {u['side']})")
            # #622 lever 2: an unsatisfied unit with NO committed copper
            # can still change sides -- the human's move for exactly these
            # nets (the conflict list matches the human board's
            # non-east exits nearly one for one).
            if not any(n in lp.fixed and n not in assumed_pairs
                       for n in u['nets']):
                lp.reside_candidates.update(u['nets'])

    # --- final per-net layers (fixed outranks unit color) ----------------
    ejected: List[str] = []
    for i, u in enumerate(units):
        for nid in u['nets']:
            f = lp.fixed.get(nid)
            lay = f if f is not None else color[i]
            lp.net_layer[nid] = lay
            if f is not None and f != color[i] and u['group'] \
                    and nid not in assumed_pairs:
                ejected.append(f"{plan.net_names.get(nid, str(nid))} -> {f} "
                               f"(unit on {color[i]}, committed copper wins)")

    # Assumed (uncommitted) pairs constrained the solve but are not seeded.
    for nid in assumed_pairs:
        lp.net_layer.pop(nid, None)
        lp.fixed.pop(nid, None)

    lp.report = {
        'assigned': {plan.net_names.get(n, str(n)): l
                     for n, l in sorted(lp.net_layer.items())},
        'assumed_pairs': sorted(plan.net_names.get(n, str(n))
                                for n in assumed_pairs),
        'fixed': {plan.net_names.get(n, str(n)): l
                  for n, l in sorted(lp.fixed.items())},
        'units': [{'side': u['side'],
                   'layer': color[i],
                   'nets': [plan.net_names.get(n, str(n)) for n in u['nets']]}
                  for i, u in enumerate(units) if u['group']],
        'violations': violations,
        'reside_candidates': sorted(plan.net_names.get(n, str(n))
                                    for n in lp.reside_candidates),
        'ejected': ejected,
    }
    if verbose:
        per_layer: Dict[str, int] = {}
        for lay in lp.net_layer.values():
            per_layer[lay] = per_layer.get(lay, 0) + 1
        print(f"  Fanout layer-plan (#622): {len(lp.net_layer)} net(s) "
              f"planned {per_layer}, {len(lp.fixed)} fixed by committed "
              f"copper/pairs, {len(violations)} conflict(s), "
              f"{len(ejected)} ejected")
        for v in violations:
            print(f"    layer-plan CONFLICT: {v}")
        for e in ejected:
            print(f"    layer-plan eject: {e}")
    return lp
