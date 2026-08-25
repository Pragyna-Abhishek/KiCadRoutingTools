"""Pre-fanout probe plan (#622): plan escape directions and comb-slot order
from REALIZED routes, before any stub exists.

The route-side probe-plan (KICAD_BUS_PROBE_PLAN, single_ended_loop) learned
that how a net attaches at each end -- where, and from what direction -- is
what decides which bus formation it can join, and that this is unpredictable
from board geometry at planning time. But by the time that pass runs, fanout
has already frozen the largest formation decision of all: which side of the
BGA every net leaves from, and in what comb order. This module moves the same
realized-geometry probe BEFORE fanout (KICAD_FANOUT_PROBE_PLAN=1):

1. Solo-A* each candidate ball's net on the unfanned board (its own honest
   obstacle map, exactly like the route-side probe).
2. Per ball: the probe's departure direction becomes the escape-direction
   preference fed to find_escape_channel (upgrading #469's straight-line
   nearest-pad guess to a realized route).
3. The probes' attach points + approach directions feed detect_bus_groups --
   the SAME clique machinery routing uses -- so buses are identified before
   a single stub exists.
4. allocate_escape_sides turns the raw preferences into a CAPACITY-AWARE
   allocation: a probe's side is granted only while that side's comb has
   room (gaps x layers, the model the human fanout of allwinner DU1
   measurably respects: left 13/~16, down 23/~30), nets closest to their
   preferred side first, spill to the natural nearest side when full.
5. Each bus's realized comb order goes into the JSON plan ONLY. Fanout does
   NOT enforce it: in-field slot permutation was built, measured, and
   REMOVED -- a slot swap forces the two 45-degree jogs to cross inside the
   ball field by construction (needs a layer split the assigner can't
   always grant; DU1 escapes fell 50/56 -> 46/56, every extra drop a moved
   member or its collision victim), and the realized order wanted
   permutations far beyond adjacent-channel reach anyway (8/11 members of
   the main group). The human tolerates edge disorder (8-25% pairwise
   inversions) and untwists in the field and close-in region -- so order is
   the ROUTE step's job, and 100% ball escape is fanout's.

Every decision here derives from realized probe geometry -- no package
heuristics, no invented angle rules. Nets the probe cannot model (multipoint,
multiple balls on this footprint, probe failure) simply get no preference and
fall through to the unplanned behavior.
"""

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# Lateral order is sampled where the realized route crosses a reference line
# this far past the BGA grid edge -- far enough that the fan toward the
# destination has begun, near enough that unrelated mid-board detours don't
# dominate.
REF_PAST_EDGE_MM = 1.5



@dataclass
class FanoutProbePlan:
    # (round(x,3), round(y,3)) ball position -> 'up'/'down'/'left'/'right',
    # same key convention as escape.preferred_escape_dirs so it drops into
    # the existing preferred_dir plumbing.
    pad_dirs: Dict[Tuple[float, float], str] = field(default_factory=dict)
    # net_id -> realized centerline in mm, BGA end FIRST.
    paths_mm: Dict[int, List[Tuple[float, float]]] = field(default_factory=dict)
    groups: List[List[int]] = field(default_factory=list)
    net_names: Dict[int, str] = field(default_factory=dict)
    # net_id -> the ball's (x, y), for the allocator and order report.
    ball_pos: Dict[int, Tuple[float, float]] = field(default_factory=dict)
    probed: int = 0
    candidates: int = 0
    # Machine-readable plan for the step's JSON_SUMMARY (the handoff to the
    # route step); allocate_escape_sides and report_bus_orders fill in the
    # allocation summary and per-bus realized comb orders.
    report: dict = field(default_factory=dict)


def _quantize_dir(dx: float, dy: float) -> Optional[str]:
    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        return None
    if abs(dx) >= abs(dy):
        return 'right' if dx > 0 else 'left'
    return 'down' if dy > 0 else 'up'  # board y grows downward


def build_fanout_probe_plan(footprint, pcb_data, net_filter,
                            layers, track_width, clearance,
                            via_size, via_drill) -> Optional[FanoutProbePlan]:
    """Probe every modelable candidate ball's net and build the plan.

    Candidates are nets with exactly ONE ball on this footprint and exactly
    TWO pads in total (the 2-point span the solo probe models faithfully);
    everything else is skipped and counted, never guessed at.
    """
    from net_queries import matches_net_filter
    from routing_config import GridRouteConfig
    from obstacle_map import build_base_obstacle_map
    from single_ended_routing import route_net_with_obstacles
    from bus_detection import detect_bus_groups

    t0 = time.time()
    plan = FanoutProbePlan()

    on_fp: Dict[int, List] = {}
    for pad in footprint.pads:
        if pad.net_id <= 0 or not pad.net_name:
            continue
        if pad.net_name.lower().startswith('unconnected-'):
            continue
        if net_filter and not matches_net_filter(pad.net_name, net_filter):
            continue
        on_fp.setdefault(pad.net_id, []).append(pad)

    cand: Dict[int, object] = {}
    for nid, pads in on_fp.items():
        total = len(pcb_data.pads_by_net.get(nid, []))
        if len(pads) == 1 and total == 2:
            cand[nid] = pads[0]
    plan.candidates = len(on_fp)

    if not cand:
        return None

    cfg = GridRouteConfig(track_width=track_width, clearance=clearance,
                          via_size=via_size, via_drill=via_drill,
                          layers=list(layers))
    gs = cfg.grid_step

    override: Dict[int, Tuple[Tuple[float, float], Tuple[float, float]]] = {}
    dirs: Dict[int, Tuple[Optional[Tuple[float, float]],
                          Optional[Tuple[float, float]]]] = {}
    for nid, ball in cand.items():
        try:
            omap = build_base_obstacle_map(pcb_data, cfg, [nid])
            r = route_net_with_obstacles(pcb_data, nid, cfg, omap)
        except Exception:
            continue
        if not r or r.get('failed') or not r.get('path') \
                or len(r['path']) < 4:
            continue
        pts = [(p[0] * gs, p[1] * gs) for p in r['path']]
        # Orient the path BGA end first.
        bx, by = ball.global_x, ball.global_y
        d0 = math.hypot(pts[0][0] - bx, pts[0][1] - by)
        d1 = math.hypot(pts[-1][0] - bx, pts[-1][1] - by)
        if d1 < d0:
            pts.reverse()
        k = min(8, len(pts) - 1)
        dep = _quantize_dir(pts[k][0] - pts[0][0], pts[k][1] - pts[0][1])
        if dep is None:
            continue
        plan.paths_mm[nid] = pts
        plan.net_names[nid] = ball.net_name
        plan.ball_pos[nid] = (bx, by)
        plan.pad_dirs[(round(bx, 3), round(by, 3))] = dep

        # Detection wants attach points + unit approach vectors INTO each
        # end (route-side probe-plan convention: source = ball end here).
        def _into(i_end, i_inner):
            vx = pts[i_end][0] - pts[i_inner][0]
            vy = pts[i_end][1] - pts[i_inner][1]
            L = math.hypot(vx, vy)
            return (vx / L, vy / L) if L > 1e-9 else None
        override[nid] = (pts[0], pts[-1])
        dirs[nid] = (_into(0, k), _into(len(pts) - 1, len(pts) - 1 - k))

    plan.probed = len(plan.paths_mm)
    if not plan.paths_mm:
        return None

    groups = detect_bus_groups(pcb_data, sorted(override),
                               endpoint_override=override,
                               endpoint_dirs=dirs)
    plan.groups = [list(g.net_ids) for g in groups]

    plan.report = {
        'probed': plan.probed,
        'candidates': plan.candidates,
        'groups': [[plan.net_names.get(n, str(n)) for n in g]
                   for g in plan.groups],
        'pad_dirs': {plan.net_names[nid]: plan.pad_dirs[k]
                     for nid, p in cand.items()
                     for k in [(round(p.global_x, 3), round(p.global_y, 3))]
                     if k in plan.pad_dirs and nid in plan.net_names},
    }
    print(f"  Fanout probe-plan: {plan.probed}/{plan.candidates} net(s) "
          f"solo-probed in {time.time() - t0:.1f}s, "
          f"{len(plan.groups)} bus group(s)")
    return plan


def _lateral_key(pts: List[Tuple[float, float]], escape_dir: str,
                 grid) -> Optional[float]:
    """Lateral coordinate where the realized route crosses the reference
    line REF_PAST_EDGE_MM past the grid edge on its escape side; the path's
    far-end lateral coordinate when it never reaches that line."""
    if escape_dir == 'down':
        ref, axis, sgn = grid.max_y + REF_PAST_EDGE_MM, 1, 1
    elif escape_dir == 'up':
        ref, axis, sgn = grid.min_y - REF_PAST_EDGE_MM, 1, -1
    elif escape_dir == 'right':
        ref, axis, sgn = grid.max_x + REF_PAST_EDGE_MM, 0, 1
    else:
        ref, axis, sgn = grid.min_x - REF_PAST_EDGE_MM, 0, -1
    lat = 1 - axis
    for i in range(1, len(pts)):
        a, b = pts[i - 1], pts[i]
        if sgn * (b[axis] - ref) >= 0 > sgn * (a[axis] - ref):
            span = b[axis] - a[axis]
            t = 0.0 if abs(span) < 1e-9 else (ref - a[axis]) / span
            return a[lat] + t * (b[lat] - a[lat])
    return pts[-1][lat]


def allocate_escape_sides(plan: FanoutProbePlan, grid, channels,
                          layers, pair_assignments=None,
                          prefer_overrides=None):
    """Capacity-aware side allocation (#622 v2, the human-fanout model).

    A probe's preferred side is granted only while that side's comb has
    room -- one track per (channel, direction) per layer, the gaps x layers
    budget the human DU1 fanout measurably respects -- processing nets
    closest to their preferred side first, so geography wins slots the way
    it does in the human layout. A net whose preferred side is full spills
    to its remaining sides by distance; a net no side can take gets no
    preference at all (the engine's natural assignment and its downstream
    layer/collision machinery deal with it, typically by diving under the
    field the way the human does). Returns the pad_dirs overlay to feed
    find_escape_channel.
    """
    from collections import Counter
    from bga_fanout.grid import is_edge_pad

    h_channels = [c for c in channels if c.orientation == 'horizontal']
    v_channels = [c for c in channels if c.orientation == 'vertical']
    if not h_channels and not v_channels:
        return {}
    cap = max(1, len(layers))
    ledger: Dict[Tuple[str, float, str], int] = {}

    # Diff pairs were assigned first (assign_pair_escapes) and hold their
    # channels already: charge them, two tracks each.
    for _pid, (ch, esc) in (pair_assignments or {}).items():
        if ch is not None:
            k = (ch.orientation, round(ch.position, 2), esc)
            ledger[k] = ledger.get(k, 0) + 2

    def _side_dist(bx, by, side):
        if side == 'left':
            return bx - grid.min_x
        if side == 'right':
            return grid.max_x - bx
        if side == 'up':
            return by - grid.min_y
        return grid.max_y - by

    def _nearest_channel(bx, by, side):
        if side in ('left', 'right'):
            return min(h_channels, key=lambda c: abs(c.position - by)) \
                if h_channels else None
        return min(v_channels, key=lambda c: abs(c.position - bx)) \
            if v_channels else None

    items = []
    for nid, (bx, by) in plan.ball_pos.items():
        # #622 lever 2: a wall-aware caller may override a net's preferred
        # side (the layer-plan's conflicted units); the grant-or-natural
        # doctrine below is unchanged -- an override is granted only while
        # that side's comb has room, else the net stays natural.
        side = (prefer_overrides or {}).get((round(bx, 3), round(by, 3))) \
            or plan.pad_dirs.get((round(bx, 3), round(by, 3)))
        if side is None:
            continue
        edge, _ = is_edge_pad(bx, by, grid)
        if edge:
            continue  # fixed outward escape, consumes no channel
        items.append((_side_dist(bx, by, side), nid, bx, by, side))
    items.sort()

    # The allocator only ever does one of two things: grant the probe's
    # preferred side while that side's comb has room, or stay SILENT (the
    # engine's natural nearest-side assignment, whose feasibility its own
    # collision/layer machinery decides). It never redirects a net to a
    # side the probe didn't ask for: a first version spill-ranked full-side
    # nets onto "next best" sides and DROPPED DU1 escapes 50/56 -> 44/56 --
    # a side-count ledger cannot out-model the engine's real geometric
    # feasibility, so forcing is always wrong. Silent nets still charge
    # their natural side so later grants see the true load.
    alloc: Dict[Tuple[float, float], str] = {}
    granted = {'preferred': 0, 'natural': 0}
    for _, nid, bx, by, pref in items:
        natural = min(('left', 'right', 'up', 'down'),
                      key=lambda s: _side_dist(bx, by, s))
        placed = None
        ch = _nearest_channel(bx, by, pref)
        if ch is not None:
            k = (ch.orientation, round(ch.position, 2), pref)
            if pref == natural or ledger.get(k, 0) < cap:
                ledger[k] = ledger.get(k, 0) + 1
                placed = pref
        if placed is None:
            nch = _nearest_channel(bx, by, natural)
            if nch is not None:
                nk = (nch.orientation, round(nch.position, 2), natural)
                ledger[nk] = ledger.get(nk, 0) + 1
            granted['natural'] += 1
            continue
        alloc[(round(bx, 3), round(by, 3))] = placed
        granted['preferred'] += 1
    plan.report['side_alloc'] = {
        'granted_preferred': granted['preferred'],
        'left_natural': granted['natural'],
        'sides': dict(Counter(alloc.values())),
    }
    print(f"  Fanout probe-plan: side allocation -- "
          f"{granted['preferred']} granted preferred side, "
          f"{granted['natural']} left natural (comb capacity); "
          f"granted sides {dict(Counter(alloc.values()))}")
    return alloc


def report_bus_orders(plan: FanoutProbePlan, grid):
    """Fill the JSON plan with each bus's realized comb order per side --
    information for the ROUTE step, deliberately not enforced by fanout
    (see the module docstring for the measured reason)."""
    orders = []
    for group in plan.groups:
        by_side: Dict[str, List[Tuple[float, str]]] = {}
        for nid in group:
            pos = plan.ball_pos.get(nid)
            if pos is None:
                continue
            side = plan.pad_dirs.get((round(pos[0], 3), round(pos[1], 3)))
            pts = plan.paths_mm.get(nid)
            if side is None or not pts:
                continue
            key = _lateral_key(pts, side, grid)
            if key is not None:
                by_side.setdefault(side, []).append(
                    (key, plan.net_names.get(nid, str(nid))))
        for side, keyed in by_side.items():
            if len(keyed) >= 2:
                orders.append({'side': side,
                               'order': [n for _, n in sorted(keyed)]})
    plan.report['bus_orders'] = orders


