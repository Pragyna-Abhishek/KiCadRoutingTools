"""
Bus detection module for identifying groups of nets that should be routed together.

A bus is a group of nets where:
1. Source endpoints are physically clustered together
2. Target endpoints are also physically clustered together

This module detects such groups and orders the nets by physical position
for routing from the middle outward.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
import math

from kicad_parser import PCBData
from connectivity import get_net_routing_endpoints


@dataclass
class BusGroup:
    """Represents a group of nets that should be routed as a bus."""
    name: str
    net_ids: List[int] = field(default_factory=list)
    # Source and target positions per net (parallel lists with net_ids)
    source_positions: List[Tuple[float, float]] = field(default_factory=list)
    target_positions: List[Tuple[float, float]] = field(default_factory=list)
    # Which endpoint type formed the clique (determines routing direction)
    clique_endpoint: str = "source"  # "source" or "target"

    @property
    def count(self) -> int:
        return len(self.net_ids)


def detect_bus_groups(
    pcb_data: PCBData,
    net_ids: List[int],
    detection_radius: float = 2.0,
    min_nets: int = 2,
    endpoint_override: Optional[Dict[int, Tuple[Tuple[float, float],
                                                Tuple[float, float]]]] = None,
    endpoint_dirs: Optional[Dict[int, Tuple[Optional[Tuple[float, float]],
                                            Optional[Tuple[float, float]]]]]
    = None,
) -> List[BusGroup]:
    """
    Detect bus groups by finding nets where EITHER all sources are within radius
    of each other OR all targets are within radius of each other.

    Args:
        pcb_data: PCB data with net information
        net_ids: List of net IDs to analyze for bus grouping
        detection_radius: Maximum distance (mm) - either all sources or all targets
                         must be within this distance of each other
        min_nets: Minimum number of nets to form a bus (default 2)

    Returns:
        List of BusGroup objects, each containing nets that form a bus
    """
    # Get endpoints for each net. #622 (take2 port, minimal): a caller with
    # REALIZED attach geometry (the fanout probe plan's solo-A* endpoints)
    # may override the derived endpoints per net; `endpoint_dirs` (approach
    # vectors) is accepted for interface parity with the probe plan but not
    # consumed by this detection -- the take-1 dirs-aware clique refinement
    # was deliberately NOT carried.
    del endpoint_dirs
    net_endpoints: Dict[int, Tuple[Tuple[float, float], Tuple[float, float]]] = {}

    for net_id in net_ids:
        if endpoint_override and net_id in endpoint_override:
            net_endpoints[net_id] = endpoint_override[net_id]
            continue
        endpoints = get_net_routing_endpoints(pcb_data, net_id)
        if len(endpoints) >= 2:
            net_endpoints[net_id] = (endpoints[0], endpoints[1])

    # Slot canonicalization (#622, take-2 -- found by the take-1 K=4
    # microscope and REDISCOVERED at take-2 K=2): the (source, target) ORDER
    # coming out of get_net_routing_endpoints is per-net arbitrary (its
    # Case-1 ranks stub groups by copper size), so neighboring nets of one
    # physical bus can arrive with opposite orientation. The cliquer
    # compares slot-to-slot, so mixed orientation scatters "sources" across
    # both ends of the bus: detection finds NOTHING at K=2 (SDQ4 U1-first,
    # SDQ1 DU1-first -- measured: no group, no attraction, two independent
    # routes), and silently fragments larger sets into orientation-pure
    # subsets. Orientation is DENSITY-based (user rule): source = the end
    # in the denser pad field, so routing starts from the constrained
    # (fanned BGA) side and escapes toward the forgiving side. Ties fall
    # back to the lexicographically smaller endpoint -- deterministic
    # either way, no thresholds.
    _all_pads = [p for fp in pcb_data.footprints.values() for p in fp.pads]
    _dens_cache: Dict[Tuple[float, float], int] = {}

    def _pad_density(pt):
        k = (round(pt[0], 1), round(pt[1], 1))
        if k not in _dens_cache:
            _dens_cache[k] = sum(
                1 for p in _all_pads
                if abs(p.global_x - pt[0]) <= 3.0
                and abs(p.global_y - pt[1]) <= 3.0)
        return _dens_cache[k]

    for _nid, (_sa, _sb) in list(net_endpoints.items()):
        _da, _db = _pad_density(_sa), _pad_density(_sb)
        if _db > _da or (_da == _db
                         and (_sb[0], _sb[1]) < (_sa[0], _sa[1])):
            net_endpoints[_nid] = (_sb, _sa)

    if len(net_endpoints) < min_nets:
        return []

    bus_groups = []
    bus_counter = 0
    remaining = set(net_endpoints.keys())

    while len(remaining) >= min_nets:
        source_positions = {nid: net_endpoints[nid][0] for nid in remaining}
        target_positions = {nid: net_endpoints[nid][1] for nid in remaining}

        # Find largest clique from sources OR targets
        source_clique = _find_largest_clique(source_positions, detection_radius, min_nets)
        target_clique = _find_largest_clique(target_positions, detection_radius, min_nets)

        # Use whichever is larger, and track which endpoint formed the clique
        if len(source_clique) >= len(target_clique):
            best_bus_nets = source_clique
            clique_endpoint = "source"
        else:
            best_bus_nets = target_clique
            clique_endpoint = "target"

        if len(best_bus_nets) >= min_nets:
            # Both-end coherence (#296 R9): a one-ended clique at a BGA sweeps
            # up EVERYTHING fanning out of the package (ottercast: a 21-net
            # "bus" mixing the SDC0 river with power nets, local caps and LEDs
            # bound for different corners -- one planned corridor for all of
            # them is noise for most). Sub-cluster the clique by the OTHER
            # endpoint (greedy centroid clustering at 2x the detection radius,
            # destinations spread wider than sources along connectors) and
            # emit only destination-coherent subgroups; the leftovers are not
            # buses and route normally.
            other = 1 if clique_endpoint == "source" else 0
            clusters: List[List[int]] = []
            for nid in best_bus_nets:
                px, py = net_endpoints[nid][other]
                placed = False
                for cl in clusters:
                    # math.fsum, not the builtin sum() (#493): Python 3.12
                    # switched sum() to compensated summation, so a float sum
                    # differs by an ULP or two between KiCad's bundled python
                    # and the system one. These centroids are compared against
                    # a radius below, so an ULP can flip cluster membership and
                    # change which nets are treated as a bus -- i.e. the routing
                    # would depend on the interpreter. fsum is exactly rounded
                    # on every version.
                    cx = math.fsum(net_endpoints[m][other][0] for m in cl) / len(cl)
                    cy = math.fsum(net_endpoints[m][other][1] for m in cl) / len(cl)
                    if math.hypot(px - cx, py - cy) <= 2.0 * detection_radius:
                        cl.append(nid)
                        placed = True
                        break
                if not placed:
                    clusters.append([nid])

            for cl in clusters:
                if len(cl) < min_nets:
                    continue
                bus_counter += 1
                bus = BusGroup(name=f"bus_{bus_counter}",
                               clique_endpoint=clique_endpoint)
                # Order nets by physical position
                ordered_nets = _order_nets_by_position(cl, net_endpoints)
                for net_id in ordered_nets:
                    bus.net_ids.append(net_id)
                    bus.source_positions.append(net_endpoints[net_id][0])
                    bus.target_positions.append(net_endpoints[net_id][1])
                bus_groups.append(bus)

            # Remove EVERY clique member from consideration (emitted or not:
            # re-considering the leftovers would just re-form the same clique).
            for nid in best_bus_nets:
                remaining.discard(nid)
        else:
            # No valid bus found, done
            break

    return bus_groups


def _find_largest_clique(
    positions: Dict[int, Tuple[float, float]],
    radius: float,
    min_size: int
) -> List[int]:
    """
    Find the largest group where all members are within radius of each other.

    Uses greedy approach: start with closest pair, add items that are within
    radius of all existing members.
    """
    if len(positions) < min_size:
        return []

    items = list(positions.keys())

    # Find all pairs within radius
    edges = []
    for i, id1 in enumerate(items):
        x1, y1 = positions[id1]
        for id2 in items[i+1:]:
            x2, y2 = positions[id2]
            dist = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
            if dist <= radius:
                edges.append((dist, id1, id2))

    if not edges:
        return []

    # Sort by distance (closest first)
    edges.sort()

    # Try building cliques starting from each edge
    best_clique = []
    for _, id1, id2 in edges:
        clique = [id1, id2]

        # Try adding other items
        for other in items:
            if other in clique:
                continue
            ox, oy = positions[other]
            # Check if within radius of all clique members
            all_close = True
            for member in clique:
                mx, my = positions[member]
                if math.sqrt((ox - mx) ** 2 + (oy - my) ** 2) > radius:
                    all_close = False
                    break
            if all_close:
                clique.append(other)

        if len(clique) > len(best_clique):
            best_clique = clique

    return best_clique if len(best_clique) >= min_size else []


def _order_nets_by_position(
    net_ids: List[int],
    net_endpoints: Dict[int, Tuple[Tuple[float, float], Tuple[float, float]]]
) -> List[int]:
    """
    Order nets by physical position (left-to-right or top-to-bottom).

    Determines the primary axis of the bus (horizontal or vertical) and
    sorts nets accordingly.

    Args:
        net_ids: List of net IDs to order
        net_endpoints: Dict mapping net ID to (source, target) positions

    Returns:
        Ordered list of net IDs
    """
    if len(net_ids) <= 1:
        return list(net_ids)

    # Get source positions for all nets
    sources = [(nid, net_endpoints[nid][0]) for nid in net_ids]

    # Determine primary axis by looking at spread in X vs Y
    xs = [p[0] for _, p in sources]
    ys = [p[1] for _, p in sources]

    x_spread = max(xs) - min(xs)
    y_spread = max(ys) - min(ys)

    # Sort by the axis with larger spread to get physical ordering
    if x_spread >= y_spread:
        # Sort by X (left to right)
        sources.sort(key=lambda item: item[1][0])
    else:
        # Sort by Y (top to bottom)
        sources.sort(key=lambda item: item[1][1])

    return [nid for nid, _ in sources]


def get_bus_routing_order(bus: BusGroup) -> List[int]:
    """
    Get the order in which bus nets should be routed.

    Routes from the middle outward, alternating sides.
    Example for 5 nets [A, B, C, D, E] ordered by position:
    Route order: [C, B, D, A, E] (middle, left, right, left, right)

    Args:
        bus: BusGroup with nets ordered by physical position

    Returns:
        List of net IDs in routing order
    """
    n = len(bus.net_ids)
    if n == 0:
        return []
    if n == 1:
        return list(bus.net_ids)

    # #622 take-2 (KICAD_BUS_FOLLOW_LEADER): DEEPEST-TARGET-FIRST. The
    # terminal plan lands members in target-depth order, and that only
    # works if the deep member routes before the shallow one -- measured:
    # with the shallow sibling routed first, its landing copper + ball
    # blocked the deep member's surface path and forced a 2-via tunnel
    # (e3_2 SDQ0); deep-first, the shallow member lands short of the
    # deep one's copper for free. Also how a human fills a comb.
    import os as _do_os
    if _do_os.environ.get('KICAD_BUS_FOLLOW_LEADER') not in (None, '', '0'):
        import math as _do_m
        n_ = float(n)
        scx = sum(p[0] for p in bus.source_positions) / n_
        scy = sum(p[1] for p in bus.source_positions) / n_
        tcx = sum(p[0] for p in bus.target_positions) / n_
        tcy = sum(p[1] for p in bus.target_positions) / n_
        tvx, tvy = tcx - scx, tcy - scy
        L = _do_m.hypot(tvx, tvy) or 1.0
        tvx, tvy = tvx / L, tvy / L
        return [nid for _, nid in sorted(
            ((bus.target_positions[i][0] * tvx
              + bus.target_positions[i][1] * tvy, nid)
             for i, nid in enumerate(bus.net_ids)),
            reverse=True)]

    middle = n // 2
    order = [bus.net_ids[middle]]

    for i in range(1, n):
        left_idx = middle - i
        right_idx = middle + i

        if left_idx >= 0:
            order.append(bus.net_ids[left_idx])
        if right_idx < n:
            order.append(bus.net_ids[right_idx])

    return order


def get_attraction_neighbor(
    bus: BusGroup,
    net_id: int,
    routed_paths: Dict[int, List[Tuple[int, int, int]]]
) -> Optional[List[Tuple[int, int, int]]]:
    """
    Get the path of the already-routed neighbor that this net should attract to.

    Args:
        bus: BusGroup containing the net
        net_id: Net ID being routed
        routed_paths: Dict mapping net ID to routed path [(gx, gy, layer), ...]

    Returns:
        Path of the neighbor to attract to, or None if no neighbor routed yet
    """
    if net_id not in bus.net_ids:
        return None

    idx = bus.net_ids.index(net_id)

    # Check left neighbor first
    if idx > 0:
        left_neighbor = bus.net_ids[idx - 1]
        if left_neighbor in routed_paths:
            return routed_paths[left_neighbor]

    # Check right neighbor
    if idx < len(bus.net_ids) - 1:
        right_neighbor = bus.net_ids[idx + 1]
        if right_neighbor in routed_paths:
            return routed_paths[right_neighbor]

    return None


def filter_bus_groups_geometric(pcb_data, bus_groups, config,
                                min_members: int = 2,
                                min_run_mm: float = 5.0,
                                cos_cone: float = 0.9,
                                max_len_ratio: float = 1.5):
    """Strict geometric bus definition (default on; KICAD_BUS_STRICT=0
    reverts to raw clique detection): a group is a bus only if its members
    genuinely TRAVEL TOGETHER. Five rules, no names, no component identity:

      1. >=2 members (#622 take2: a 2-net co-traveling pair IS a bus --
         the human packs pairs too, and the K=2 microscope is the
         campaign's acceptance test; was 3 -- short/incoherent pairs are
         still killed by rules 2-5),
      2. parallel travel: member displacement vectors within a cos>0.9
         cone at length ratio <=1.5 (largest coherent subset kept --
         clustered sources + this cone geometrically imply clustered
         destinations, subsuming any shared-component test),
      3. run length >= 5mm (local hops gain nothing; also retires the
         decoupling-cap clusters without a passive-component carve-out),
      4. no power nets (config.power_net_widths),
      5. no diff-pair members (config-independent: *_P/*_N by pair map is
         the caller's concern; here power/width classes only).

    Raw clique detection on ottercast declared 31 groups/118 members and
    planned corridors for decoupling clusters and power rails; these rules
    yield 6 groups/27 members -- the four real buses (SDC0, WL_SDIO,
    BT_PCM, BT_UART) plus small co-traveling GPIO/cap bundles a human
    would also lane together.
    """
    import os as _os
    if _os.environ.get('KICAD_BUS_STRICT', '1') in ('0', 'off', 'false'):
        return bus_groups
    import math as _math
    power_ids = set(getattr(config, 'power_net_widths', {}) or {})
    out = []
    for bus in bus_groups:
        vecs = {}
        for nid in bus.net_ids:
            if nid in power_ids:
                continue
            pads = pcb_data.pads_by_net.get(nid, [])
            if len(pads) < 2:
                continue
            a = min(pads, key=lambda p: p.global_x + p.global_y)
            b = max(pads, key=lambda p: p.global_x + p.global_y)
            dx, dy = b.global_x - a.global_x, b.global_y - a.global_y
            L = _math.hypot(dx, dy)
            if L >= min_run_mm:
                vecs[nid] = (dx / L, dy / L, L)
        best = []
        for nid, (ux, uy, L) in vecs.items():
            cone = [m for m, (vx, vy, M) in vecs.items()
                    if ux * vx + uy * vy > cos_cone
                    and max(L, M) / min(L, M) <= max_len_ratio]
            if len(cone) > len(best):
                best = cone
        if len(best) < min_members:
            continue
        if len(best) < len(bus.net_ids):
            # Prune the PARALLEL lists together: net_ids alone leaves
            # source/target_positions misaligned by one slot per pruned
            # member, and every downstream consumer that indexes
            # positions by net_ids.index() then reads a NEIGHBOR's
            # geometry (measured h8_15: SDQ11 planned its landing at
            # pruned SA12's ball; 11 of 14 members got no berth). Same
            # disease as the 835b6dd6 position-sync fix on the take-1
            # branch.
            bset = set(best)
            keep_idx = [i for i, nid in enumerate(bus.net_ids)
                        if nid in bset]
            if len(bus.source_positions) == len(bus.net_ids):
                bus.source_positions = [bus.source_positions[i]
                                        for i in keep_idx]
            if len(bus.target_positions) == len(bus.net_ids):
                bus.target_positions = [bus.target_positions[i]
                                        for i in keep_idx]
            bus.net_ids = [bus.net_ids[i] for i in keep_idx]
        out.append(bus)
    return out


def net_teeth_common_layer(pcb_data, net_id) -> Optional[str]:
    """The single layer shared by ALL of the net's pre-existing free stub
    tips (degree-1 segment endpoints not anchored at same-net pads/vias),
    or None when the net has no stubs or its tips span layers. Board
    geometry is the honest source (#622: published plans lied twice); a
    via-in-pad + stub tooth reads as its stub's layer -- exactly the layer
    a trunk to this net must ride to attach."""
    from collections import Counter as _Counter
    segs = [s for s in pcb_data.segments if s.net_id == net_id]
    if not segs:
        return None
    deg = _Counter()
    for s in segs:
        deg[(round(s.start_x, 3), round(s.start_y, 3))] += 1
        deg[(round(s.end_x, 3), round(s.end_y, 3))] += 1
    # Tight anchor radius: fanout-authored geometry is exact, and a short
    # tooth's real tip can sit < 0.1mm past its own dog-bone via (take-1
    # measured: 0.15 swallowed a 0.088mm tip and returned None for exactly
    # the net that needed the answer).
    anchors = [(p.global_x, p.global_y)
               for p in pcb_data.pads_by_net.get(net_id, [])]
    anchors += [(v.x, v.y) for v in pcb_data.vias if v.net_id == net_id]

    def _anchored(x, y):
        return any((x - ax) ** 2 + (y - ay) ** 2 < 0.05 ** 2
                   for ax, ay in anchors)
    tip_layers = set()
    n_tips = 0
    for s in segs:
        for pt in ((round(s.start_x, 3), round(s.start_y, 3)),
                   (round(s.end_x, 3), round(s.end_y, 3))):
            if deg[pt] == 1 and not _anchored(pt[0], pt[1]):
                tip_layers.add(s.layer)
                n_tips += 1
    if n_tips < 2 or len(tip_layers) != 1:
        return None
    return next(iter(tip_layers))


def net_end_attach_layers(pcb_data, net_id, radius=2.5):
    """Per-terminal ATTACH-LAYER sets: for each pad of the net, the
    copper layers a trunk can reach that terminal on without adding a
    via. A stub near the pad contributes its layer; a same-net VIA near
    the pad (a dogbone) contributes its whole span -- the via is a free
    layer change the router already owns; a bare pad contributes its
    own copper layers (a BGA ball = the surface layer only).

    This is the terminal truth net_teeth_common_layer cannot see on a
    dogbone fanout (#622 K=30 microscope: pad->stub->via has NO free
    tip, so every tooth read None and the macro-river max-cut priced
    terminal reality at ZERO -- it assigned 9 members to a B river
    whose every member pays an unpriced dive at its F-locked ball end,
    and 8 of the 9 defected)."""
    pads = pcb_data.pads_by_net.get(net_id, [])
    segs = [s for s in pcb_data.segments if s.net_id == net_id]
    vias = [v for v in pcb_data.vias if v.net_id == net_id]
    copper = set(pcb_data.board_info.copper_layers or [])
    r2 = radius * radius
    out = []
    for p in pads:
        px, py = p.global_x, p.global_y
        lays = set()
        for s in segs:
            if (s.start_x - px) ** 2 + (s.start_y - py) ** 2 <= r2 or \
                    (s.end_x - px) ** 2 + (s.end_y - py) ** 2 <= r2:
                lays.add(s.layer)
        for v in vias:
            if (v.x - px) ** 2 + (v.y - py) ** 2 <= r2:
                span = set(v.layers or []) & copper
                lays.update(span if span else copper)
        if not lays:
            if getattr(p, 'drill', 0) and p.pad_type != 'np_thru_hole':
                lays = set(copper)
            else:
                lays = {l for l in (p.layers or []) if l in copper}
        out.append(lays)
    return out


def replan_groups_from_board(pcb_data, bus_groups):
    """#622 take2 (user design, KICAD_BUS_PLAN_FROM_BOARD): re-derive each
    detected group's bus structure at ROUTE time from the COMMITTED board --
    the real fanout teeth -- instead of anything planned before the fanout
    ran. Members partition by their MEASURED tooth layer (a member with no
    or mixed-layer teeth becomes a singleton and routes plain), and each
    layer class becomes its own group ordered along the entry comb (the
    projection of member positions on the perpendicular of the group's
    travel direction). Layer-homogeneous buses in measured comb order, by
    construction. Returns (new_groups, rank_map) where rank_map is
    net_id -> (chain, rank, layer), stashed on pcb_data._board_bus_rank."""
    import math as _m
    out = []
    rank_map = {}
    chain_no = 0
    for g in bus_groups:
        lay_of = {nid: net_teeth_common_layer(pcb_data, nid)
                  for nid in g.net_ids}
        classes: Dict[Optional[str], List[int]] = {}
        for k, nid in enumerate(g.net_ids):
            classes.setdefault(lay_of[nid], []).append(k)
        n = len(g.net_ids)
        sc = (sum(p[0] for p in g.source_positions) / n,
              sum(p[1] for p in g.source_positions) / n)
        tc = (sum(p[0] for p in g.target_positions) / n,
              sum(p[1] for p in g.target_positions) / n)
        tv = (tc[0] - sc[0], tc[1] - sc[1])
        L = _m.hypot(*tv) or 1.0
        perp = (-tv[1] / L, tv[0] / L)
        for lay, ks in sorted(classes.items(),
                              key=lambda t: (t[0] is None, str(t[0]))):
            if lay is None:
                for k in ks:
                    out.append(BusGroup(
                        name=f"{g.name}m{chain_no}",
                        net_ids=[g.net_ids[k]],
                        source_positions=[g.source_positions[k]],
                        target_positions=[g.target_positions[k]],
                        clique_endpoint=g.clique_endpoint))
                    chain_no += 1
                continue
            ks_sorted = sorted(ks, key=lambda k: (
                g.source_positions[k][0] * perp[0]
                + g.source_positions[k][1] * perp[1]))
            ng = BusGroup(
                name=f"{g.name}c{chain_no}",
                net_ids=[g.net_ids[k] for k in ks_sorted],
                source_positions=[g.source_positions[k] for k in ks_sorted],
                target_positions=[g.target_positions[k] for k in ks_sorted],
                clique_endpoint=g.clique_endpoint)
            out.append(ng)
            for r, k in enumerate(ks_sorted):
                rank_map[g.net_ids[k]] = (chain_no, r, lay)
            chain_no += 1
    setattr(pcb_data, '_board_bus_rank', rank_map)
    return out, rank_map


def bus_stick_config(config, attraction_path):
    """Off-lane surcharge (the STICK, KICAD_BUS_OFFLANE_MULT, default 1.0
    = off): a bus member with a planned corridor routes with every layer's
    step cost scaled by the multiplier; the corridor attraction discount
    then makes the LANE the only normal-priced place on the board. Without
    this the attraction is carrot-only -- a member whose direct area is
    congested hops outside the attraction radius, where the corridor
    exerts zero pull, and defects to a lone detour for free (WL_SDIO_D1's
    8mm northern flight while its lane ran direct at y~94). Forbidden
    layers (cost -1) stay forbidden; the member's config is otherwise
    untouched (via cost, clearances, caches all nominal)."""
    import env_knobs
    m = env_knobs.BUS_OFFLANE_MULT
    if m == 1.0 or not attraction_path:
        return config
    from dataclasses import replace as _replace
    base = list(config.layer_costs or []) or [1.0] * len(config.layers)
    while len(base) < len(config.layers):
        base.append(1.0)
    scaled = [c if c < 0 else c * m for c in base]
    return _replace(config, layer_costs=scaled)


def terminal_plan_target(bg: BusGroup, net_id: int,
                         leader_path: List[Tuple[int, int, int]],
                         config, pcb_data=None
                         ) -> Optional[List[Tuple[int, int, int]]]:
    """#622 take-2 terminal approach plan (user design, polarity-aware).

    The follower's attraction target is not the leader's path itself but
    the leader's REALIZED path offset one pack pitch onto the follower's
    LANE SIDE, truncated at the peel-off point, plus a straight landing
    leg to the follower's own target ball. The lane side is chosen by the
    TARGET (the side of the leader's path the follower's ball lies on
    near the terminal), so the destination end is clean by construction
    -- measured without this: both targets in one ball column, the pack's
    lane order inverted against target depth, and the deep member paid 2
    terminal vias tunneling around its sibling's ball (e1_2, SDQ0).

    POLARITY: when the follower's SOURCE lies on the other side of the
    leader than its target, the crossing is topologically forced (Jordan
    -- no lane assignment removes it); the offset then ramps in over the
    first ~2mm so the swap happens AT THE MOUTH, where teeth are spread
    and it is cheap, never mid-span and never at the destination.
    Returns None when the geometry is degenerate (target = attraction
    stays the raw leader path)."""
    import math as _m
    if not leader_path or len(leader_path) < 4 \
            or net_id not in bg.net_ids:
        return None
    gs = config.grid_step
    k = bg.net_ids.index(net_id)
    src = bg.source_positions[k]
    tgt = bg.target_positions[k]
    sxy = (src[0] / gs, src[1] / gs)
    txy = (tgt[0] / gs, tgt[1] / gs)
    pts = list(leader_path)
    # Orient the leader path so its END is the terminal (nearer the
    # member's target) -- routed path orientation is not guaranteed.
    d0 = (pts[0][0] - txy[0]) ** 2 + (pts[0][1] - txy[1]) ** 2
    d1 = (pts[-1][0] - txy[0]) ** 2 + (pts[-1][1] - txy[1]) ** 2
    if d0 < d1:
        pts = pts[::-1]
    peel = min(range(len(pts)),
               key=lambda i: (pts[i][0] - txy[0]) ** 2
               + (pts[i][1] - txy[1]) ** 2)
    # Back the peel off ~1.5 ball pitches so the gap leg owns the final
    # approach (same rule as the leader's terminal refinement).
    _bp = local_ball_pitch(pcb_data, tgt) if pcb_data is not None else 0.0
    if _bp > 0:
        _back = 1.5 * _bp / gs
        while peel > 3 and ((pts[peel][0] - txy[0]) ** 2
                            + (pts[peel][1] - txy[1]) ** 2) < _back * _back:
            peel -= 1
    if peel < 3:
        return None

    def _tang(i):
        a = pts[max(0, i - 3)]
        b = pts[min(len(pts) - 1, i + 3)]
        vx, vy = b[0] - a[0], b[1] - a[1]
        L = _m.hypot(vx, vy) or 1.0
        return vx / L, vy / L

    # Lane side = side of the leader the follower's BALL lies on at the
    # peel point (left-normal dot); source side likewise at the start.
    txp, typ = _tang(peel)
    _st = (txy[0] - pts[peel][0]) * (-typ) + (txy[1] - pts[peel][1]) * txp
    s_t = 1.0 if _st >= 0 else -1.0
    tx0, ty0 = _tang(0)
    _ss = (sxy[0] - pts[0][0]) * (-ty0) + (sxy[1] - pts[0][1]) * tx0
    s_s = 1.0 if _ss >= 0 else -1.0
    pitch = max(1.0, (config.track_width + config.clearance) / gs)
    # Polarity mismatch -> longer ramp: the forced swap lands at the mouth.
    ramp = 20 if s_s != s_t else 8
    out: List[Tuple[int, int, int]] = []
    for i in range(peel + 1):
        txi, tyi = _tang(i)
        f = min(1.0, i / float(ramp))
        off = pitch * s_t * f
        out.append((int(round(pts[i][0] - tyi * off)),
                    int(round(pts[i][1] + txi * off)),
                    pts[i][2]))
    _corners: List[Tuple[int, int]] = []
    _append_landing_leg(out, txy, config, pcb_data,
                        ball_pitch_mm=_bp, corners_out=_corners)
    return out, _corners


def _append_landing_leg(out, txy, config, pcb_data=None,
                        ball_pitch_mm: float = 0.0,
                        corners_out: Optional[List[Tuple[int, int]]] = None):
    """Gap-aware landing leg (#622 take-2): from the current target end,
    run LATERALLY to the ball's gap column (half a ball pitch to the side
    nearer the approach), then AXIS-ALIGNED along the gap to the ball's
    row, then jog onto the ball -- the human's straight column run. A
    diagonal corner-cut leg crosses the neighboring ball's face and
    walls the sibling that lands there (measured: e4_2 SDQ2 paid 2 vias
    hopping around the leader's corner-cut). Falls back to the straight
    leg when no pitch is known."""
    lx, ly, ll = out[-1]
    gp = ball_pitch_mm / config.grid_step if ball_pitch_mm > 0 else 0.0
    dx, dy = txy[0] - lx, txy[1] - ly
    if gp >= 2.0:
        # entry axis = dominant approach component; lateral = the other
        if abs(dy) >= abs(dx):
            gx_ = txy[0] - (gp / 2.0 if lx <= txy[0] else -gp / 2.0)
            wp = [(gx_, ly), (gx_, txy[1])]
        else:
            gy_ = txy[1] - (gp / 2.0 if ly <= txy[1] else -gp / 2.0)
            wp = [(lx, gy_), (txy[0], gy_)]
        pts = [(lx, ly)] + wp + [txy]
        if corners_out is not None:
            corners_out.extend((int(round(px_)), int(round(py_)))
                               for px_, py_ in wp)
    else:
        pts = [(lx, ly), txy]
    for (ax_, ay_), (bx_, by_) in zip(pts, pts[1:]):
        steps = max(1, int(max(abs(bx_ - ax_), abs(by_ - ay_))))
        for j in range(1, steps + 1):
            out.append((int(round(ax_ + (bx_ - ax_) * j / steps)),
                        int(round(ay_ + (by_ - ay_) * j / steps)), ll))


def local_ball_pitch(pcb_data, tgt_mm, component_hint: float = 1.5) -> float:
    """Nearest same-board pad-to-pad distance around a target (mm) -- the
    ball grid pitch the landing leg threads. 0.0 when nothing is near."""
    import math as _m
    best = 0.0
    cand = []
    for fp in pcb_data.footprints.values():
        for p in fp.pads:
            d = _m.hypot(p.global_x - tgt_mm[0], p.global_y - tgt_mm[1])
            if d < component_hint:
                cand.append((d, p))
    cand.sort(key=lambda t: t[0])
    if len(cand) >= 2 and cand[0][0] < 0.2:
        best = _m.hypot(cand[1][1].global_x - cand[0][1].global_x,
                        cand[1][1].global_y - cand[0][1].global_y)
    return best


def terminal_leader_target(bg: BusGroup, net_id: int,
                           corridor_path: List[Tuple[int, int, int]],
                           config, pcb_data
                           ) -> Optional[List[Tuple[int, int, int]]]:
    """#622 take-2: the LEADER's terminal refinement -- its corridor
    truncated at the peel-off point plus the gap-aware landing leg (no
    lateral offset; the leader owns the centerline). Keeps the leader
    from corner-cutting into the ball field across its siblings' faces."""
    if not corridor_path or len(corridor_path) < 4 \
            or net_id not in bg.net_ids:
        return None
    gs = config.grid_step
    k = bg.net_ids.index(net_id)
    tgt = bg.target_positions[k]
    txy = (tgt[0] / gs, tgt[1] / gs)
    pts = list(corridor_path)
    d0 = (pts[0][0] - txy[0]) ** 2 + (pts[0][1] - txy[1]) ** 2
    d1 = (pts[-1][0] - txy[0]) ** 2 + (pts[-1][1] - txy[1]) ** 2
    if d0 < d1:
        pts = pts[::-1]
    peel = min(range(len(pts)),
               key=lambda i: (pts[i][0] - txy[0]) ** 2
               + (pts[i][1] - txy[1]) ** 2)
    # Back the peel off to ~1.5 ball pitches from the target so the leg
    # (not the corridor tail) owns the final approach.
    bp = local_ball_pitch(pcb_data, tgt)
    if bp > 0:
        back = 1.5 * bp / gs
        while peel > 3 and ((pts[peel][0] - txy[0]) ** 2
                            + (pts[peel][1] - txy[1]) ** 2) < back * back:
            peel -= 1
    if peel < 3:
        return None
    out = list(pts[:peel + 1])
    _corners: List[Tuple[int, int]] = []
    _append_landing_leg(out, txy, config, pcb_data, ball_pitch_mm=bp,
                        corners_out=_corners)
    return out, _corners


def bus_attraction_context(
    net_id: int,
    bus_net_to_group: Optional[Dict[int, BusGroup]],
    bus_corridors: Optional[Dict[str, List[Tuple[int, int, int]]]],
    routed_paths: Optional[Dict[int, List[Tuple[int, int, int]]]] = None
) -> Tuple[Optional[List[Tuple[int, int, int]]], bool]:
    """(attraction_path, reverse_direction) for a bus member.

    The one place that encodes the priority every routing site must share:
    the group's PLANNED corridor first (one centerline for the whole group,
    guide included), else an already-routed neighbor's path; route from the
    clustered endpoints when the clique was target-based. Non-members get
    (None, False). Callers pass whatever paths dict they track (the main
    loop's sampled bus paths, the reroute loop's routed_net_paths) or None.
    """
    bg = (bus_net_to_group or {}).get(net_id)
    if bg is None:
        return None, False
    # #622 take2 D (KICAD_BUS_FOLLOW_LEADER=1): a routed sibling's REALIZED
    # path outranks the planned corridor -- the leader follows the plan,
    # every follower packs against actual copper (the human's pack; the
    # shared-centerline attraction left members 2-3mm apart because two
    # tracks cannot both occupy the centerline, measured at K=2).
    import os as _fl_os
    if _fl_os.environ.get('KICAD_BUS_FOLLOW_LEADER') not in (None, '', '0'):
        _nb = get_attraction_neighbor(bg, net_id, routed_paths or {})
        if _nb is not None:
            return _nb, (bg.clique_endpoint == "target")
    attr = (bus_corridors or {}).get(bg.name)
    if attr is not None and _fl_os.environ.get('KICAD_BUS_RIBBON'):
        # #622 ribbon: the member gets its OWN LANE (centerline offset
        # by rank at pitch, planned dive layering at the ends) instead
        # of the shared centerline every member fights over
        attr = member_lane_path(bg, net_id, attr)
    if attr is None:
        attr = get_attraction_neighbor(bg, net_id, routed_paths or {})
    return attr, (bg.clique_endpoint == "target")


def prepare_ribbon_plan(pcb_data, config, group, path_cells):
    """#622 realism fixes 1+2: turn the group's corridor CENTERLINE into
    a per-member RIBBON plan (a lane per member at track+clearance
    pitch, ordered like the group's physical order) and plan the DIVE
    POINTS (where a member whose terminal cannot attach on the river
    layer changes layer -- encoded as the lane path's layer sequence, so
    the layer-aware attraction steers the dive instead of every member
    improvising its transition in the mouth and paying excursions;
    measured K=30: ~4 extra mid-vias per river member vs the human's
    2). Stored on the group as `_ribbon`; consumed by
    bus_attraction_context when KICAD_BUS_RIBBON=1."""
    import math as _m
    n = len(group.net_ids)
    if n < 2 or not path_cells or len(path_cells) < 2:
        return
    gs = config.grid_step
    pitch = (config.track_width + config.clearance) / gs

    def _side(pt_mm):
        cx, cy = pt_mm[0] / gs, pt_mm[1] / gs
        j = min(range(len(path_cells)),
                key=lambda i: (path_cells[i][0] - cx) ** 2
                + (path_cells[i][1] - cy) ** 2)
        j2 = min(j + 1, len(path_cells) - 1)
        j1 = max(j - 1, 0)
        tx = path_cells[j2][0] - path_cells[j1][0]
        ty = path_cells[j2][1] - path_cells[j1][1]
        return ((cx - path_cells[j][0]) * (-ty)
                + (cy - path_cells[j][1]) * tx)

    # lane-order sign: the first member's lane lands on the first
    # member's side of the centerline
    s_first = _side(group.source_positions[0])
    s_last = _side(group.source_positions[-1])
    sign = 1.0 if s_first <= s_last else -1.0
    # dive plan per member and path end: the layer a trunk can attach
    # on at that terminal without a via (net_end_attach_layers); when
    # the river layer is not in the set, the lane's terminal stretch
    # rides the attach layer and the switch point IS the planned dive
    layers = list(getattr(config, 'layers', None) or [])
    dive = {}
    for nid in group.net_ids:
        pads = pcb_data.pads_by_net.get(nid, [])
        if not pads:
            continue
        attach = net_end_attach_layers(pcb_data, nid)
        ends = []
        for e in (0, -1):
            ex, ey = (path_cells[e][0] * gs, path_cells[e][1] * gs)
            li = path_cells[e][2]
            river_nm = layers[li] if 0 <= li < len(layers) else None
            k = min(range(len(pads)),
                    key=lambda i: (pads[i].global_x - ex) ** 2
                    + (pads[i].global_y - ey) ** 2)
            aset = attach[k] if k < len(attach) else set()
            if river_nm and aset and river_nm not in aset \
                    and len(aset) == 1:
                nm = next(iter(aset))
                ends.append(layers.index(nm) if nm in layers else None)
            else:
                ends.append(None)
        if ends[0] is not None or ends[1] is not None:
            dive[nid] = tuple(ends)
    group._ribbon = {'pitch': pitch, 'sign': sign, 'dive': dive,
                     'run': max(4, int(round(0.8 / gs)))}


def member_lane_path(group, net_id, path_cells):
    """The member's OWN lane from the group ribbon plan: the corridor
    centerline offset laterally by the member's rank at pitch, with the
    planned dive layering applied at the ends. Falls back to the
    centerline when no ribbon plan exists."""
    import math as _m
    rp = getattr(group, '_ribbon', None)
    if not rp or net_id not in group.net_ids or len(path_cells) < 2:
        return path_cells
    n = len(group.net_ids)
    rank = group.net_ids.index(net_id)
    off = rp['sign'] * (rank - (n - 1) / 2.0) * rp['pitch']
    # capacity clamp: a too-wide ribbon cannot fit the necks the
    # corridor was probed through (K=51: a 15-member river's 6.3mm
    # ribbon turned the lanes into an obstacle fight -- 4 failed / 90
    # rips); outer members share the edge lanes instead. Cap 4 pitches:
    # a 9-member river (rank offsets to +-4) fits UNCLAMPED -- the
    # measured K=30 win (0 failed / 79 vias / 0 rips) needs exactly
    # that, and a 3-pitch cap that clipped it cost 89 vias / 5 rips.
    _cap = 4.0 * rp['pitch']
    off = max(-_cap, min(_cap, off))
    out = []
    for i, (x, y, l) in enumerate(path_cells):
        j2 = min(i + 1, len(path_cells) - 1)
        j1 = max(i - 1, 0)
        tx = path_cells[j2][0] - path_cells[j1][0]
        ty = path_cells[j2][1] - path_cells[j1][1]
        L = _m.hypot(tx, ty)
        if L < 1e-9:
            out.append((x, y, l))
            continue
        out.append((int(round(x - ty / L * off)),
                    int(round(y + tx / L * off)), l))
    ends = rp['dive'].get(net_id)
    if ends:
        run = rp['run']
        if ends[0] is not None:
            out[:run] = [(p[0], p[1], ends[0]) for p in out[:run]]
        if ends[1] is not None:
            out[-run:] = [(p[0], p[1], ends[1]) for p in out[-run:]]
    return out


def _principal_scalar(points):
    """Project points onto their principal axis; returns the scalar list
    (length-weighted PCA is overkill for endpoint clouds -- plain
    covariance eigenvector)."""
    import math as _m
    n = float(len(points))
    mx = sum(p[0] for p in points) / n
    my = sum(p[1] for p in points) / n
    sxx = sum((p[0] - mx) ** 2 for p in points)
    syy = sum((p[1] - my) ** 2 for p in points)
    sxy = sum((p[0] - mx) * (p[1] - my) for p in points)
    tr, det = sxx + syy, sxx * syy - sxy * sxy
    lam = tr / 2 + _m.sqrt(max(0.0, tr * tr / 4 - det))
    ax = (lam - syy, sxy) if abs(sxy) > 1e-9 else \
        ((1.0, 0.0) if sxx >= syy else (0.0, 1.0))
    L = _m.hypot(*ax) or 1.0
    ux, uy = ax[0] / L, ax[1] / L
    return [(p[0] - mx) * ux + (p[1] - my) * uy for p in points]


def _accept_repartition(pcb_data, bus_groups, nets_plan):
    """#622 part B 'accept': move ONLY the members whose round-1
    REALIZED layer contradicts their river assignment. Measured basis
    (K=30 microscope): 8 of 11 B-assigned members rode F for most of
    their mid-run (SDQ4 6% B, SA3 4%) DESPITE the armed 5x off-layer
    tax, and at 20x they either still refused (SA3 3%) or complied via
    a 48mm detour and a lost net -- their assigned-layer path is
    INFEASIBLE (river capacity: a rung-2 corridor holds ~5 lanes, the
    river had 9 members), not underpriced. A defector legitimized onto
    the layer it demonstrably used packs with that river instead of
    freeloading as off-plan copper. Unlike the (measured-harmful)
    partition mode this never touches a member whose realization AGREES
    with its assignment. Rivers re-sort by comb principal order after
    moves; a river never shrinks below 2 members."""
    import os as _ar_os
    # thresholds: a member moves only when its round-1 use of the
    # ASSIGNED layer was near-zero and its use of the other layer
    # decisive. Tight by default so a healthy round 1 no-ops (K=51
    # measured: loose 0.4/0.6 moved 11 marginal members and cost +47
    # vias -- the accepted layer saturates); K=30's catastrophic
    # defectors sat at 0%, which tight thresholds still catch.
    _lo = float(_ar_os.environ.get('KICAD_BUS_PLAN_ACCEPT_LO', '0.15'))
    _hi = float(_ar_os.environ.get('KICAD_BUS_PLAN_ACCEPT_HI', '0.8'))
    by_parent = {}
    for bus in bus_groups:
        if getattr(bus, 'river_layer', None):
            par = bus.name.rsplit('_r', 1)[0] if '_r' in bus.name \
                else bus.name
            by_parent.setdefault(par, []).append(bus)
    for par, rivers in by_parent.items():
        if len(rivers) < 2:
            continue
        lay_to_river = {r.river_layer: r for r in rivers}
        changed = set()

        for r in rivers:
            for k in reversed(range(len(r.net_ids))):
                if len(r.net_ids) <= 2:
                    break
                nid = r.net_ids[k]
                nm = (pcb_data.nets[nid].name
                      if nid in pcb_data.nets else '')
                lays = (nets_plan.get(nm) or {}).get('layers_mm') or {}
                tot = sum(lays.values())
                if tot <= 0:
                    continue
                fr_asn = lays.get(r.river_layer, 0.0) / tot
                dom = max(lays, key=lays.get)
                if fr_asn < _lo and lays[dom] / tot > _hi \
                        and dom != r.river_layer \
                        and dom in lay_to_river \
                        and lay_to_river[dom] is not r:
                    dest = lay_to_river[dom]
                    dest.net_ids.append(nid)
                    dest.source_positions.append(r.source_positions[k])
                    dest.target_positions.append(r.target_positions[k])
                    del r.net_ids[k]
                    del r.source_positions[k]
                    del r.target_positions[k]
                    changed.update((id(r), id(dest)))
                    print(f"  Bus plan (accept): {nm.split('/')[-1]} "
                          f"{r.river_layer} river -> {dom} (rode "
                          f"{r.river_layer} {fr_asn:.0%} in round 1)")
        for r in rivers:
            if id(r) not in changed or len(r.net_ids) < 2:
                continue
            combs = (r.target_positions
                     if r.clique_endpoint == "target"
                     else r.source_positions)
            s = _principal_scalar(combs)
            order = sorted(range(len(r.net_ids)), key=lambda i: s[i])
            r.net_ids = [r.net_ids[i] for i in order]
            r.source_positions = [r.source_positions[i] for i in order]
            r.target_positions = [r.target_positions[i] for i in order]
    return bus_groups


def decorate_groups_from_plan(pcb_data, bus_groups, config):
    """#622 part B, DECORATE mode (KICAD_BUS_PLAN_MODE=decorate): the
    max-cut river split keeps choosing layers and membership; the plan
    contributes only what the routed board actually knows better --
    each river's member ORDER (realized lateral offsets) and its
    CORRIDOR reference (the median member's realized path). Motivated
    by measurement: partition mode's realized-layer pinning reproduced
    the source board's layer noise (r2_51 failed 1 / 229 vias, r3_51
    failed 4 / 245 vs the unplanned 0 / 202)."""
    import os as _os
    import json as _json
    path = _os.environ.get('KICAD_BUS_PLAN_FILE')
    if not path or not bus_groups or \
            _os.environ.get('KICAD_BUS_PLAN_MODE',
                            'decorate') != 'decorate':
        return bus_groups
    try:
        with open(path, encoding='utf-8') as f:
            nets_plan = _json.load(f).get('nets', {})
    except (OSError, ValueError) as e:
        print(f"  Bus plan: FAILED to read {path}: {e} -- ignored")
        return bus_groups
    # KICAD_BUS_PLAN_LEARN selects the ingredients: 'corridor'
    # (default -- the one measured non-harmful channel: 0/207 vs
    # the unplanned 0/202 at K=51) and 'accept' (acceptance
    # repartition, see _accept_repartition), comma-combinable.
    # A realized-ORDER channel was tried and REMOVED: realized
    # order is LANDING order, but net_ids order is consumed as
    # COMB order by the berth inversion costing, so pinning it
    # mis-priced every inversion (measured 1 failed/240 vias
    # alone, and the poison in every combined arm).
    _learn = set((_os.environ.get('KICAD_BUS_PLAN_LEARN')
                  or 'corridor').split(','))
    if 'accept' in _learn:
        bus_groups = _accept_repartition(pcb_data, bus_groups,
                                         nets_plan)

    def _plen(p):
        return sum(math.hypot(b[0] - a[0], b[1] - a[1])
                   for a, b in zip(p, p[1:]))

    for bus in bus_groups:
        recs = {}
        for nid in bus.net_ids:
            nm = pcb_data.nets[nid].name if nid in pcb_data.nets else ''
            if nm in nets_plan and nets_plan[nm].get('path'):
                recs[nid] = nets_plan[nm]
        if len(recs) < 2:
            continue
        # corridor member = length-median (offset-median and
        # middle-half-shortest variants were tried and removed: neither
        # ever beat length-median on a board)
        med = sorted(recs, key=lambda nid: _plen(recs[nid]['path']))[
            len(recs) // 2]
        if 'corridor' in _learn:
            # drop the leading BALL EDGE: the first path edge is the
            # corridor member's private landing leg (ball -> tip, up
            # to ~7mm of one member's diagonal); the corridor starts
            # at the leg tip like the probe's stand-off trunk
            _cp = recs[med]['path']
            bus._plan_corridor_mm = _cp[1:] if len(_cp) > 2 else _cp
        nm = [pcb_data.nets[nid].name.split('/')[-1]
              for nid in bus.net_ids]
        print(f"  Bus plan (decorate {'+'.join(sorted(_learn))}): "
              f"{bus.name} order {' '.join(nm)}")
    return bus_groups


def split_groups_to_layer_rivers(pcb_data, bus_groups, config):
    """#622 macro river plan (user law #3, KICAD_BUS_MACRO_RIVERS=1):
    partition each bus group into DIRECTIONAL RIVERS and give each river
    ONE layer, so crossings between rivers become FREE (a layer split
    pays nothing) and only within-river twist remains paid.

    Measured basis (disorder.py vs the human original): at K=30 the
    human CROSSES MORE than we do (206 vs 106) but pays for less (73 vs
    57 same-layer, 26 vs 73 mid-run vias) -- its F/B directional rivers
    make crossings free. Our single-layer groups pay every inversion
    mid-flight.

    Mechanism: crossing graph = member pairs whose comb order inverts
    against their landing order (axis-aligned by majority so a global
    flip is not 100% crossings); 2-color it MAX-CUT-style by local
    search, cost = monochromatic crossing edges + teeth-layer mismatches
    (a mismatched tooth = the human's 2-via dive, same price as a paid
    crossing). Split only when the 2-river cost beats the whole-group
    cost; each side keeps >= 2 members. Parallel lists are sliced
    TOGETHER (the position-sync lesson). Each river carries river_layer
    for bus_river_layer_config."""
    import os as _os
    if not _os.environ.get('KICAD_BUS_MACRO_RIVERS'):
        return bus_groups
    out = []
    for bus in bus_groups:
        n = len(bus.net_ids)
        if n < 4:
            out.append(bus)
            continue
        if bus.clique_endpoint == "target":
            combs, lands = bus.target_positions, bus.source_positions
        else:
            combs, lands = bus.source_positions, bus.target_positions
        comb_s = _principal_scalar(combs)
        land_s = _principal_scalar(lands)
        # align landing axis to comb order by pair majority
        agree = sum(1 for i in range(n) for j in range(i + 1, n)
                    if (comb_s[i] - comb_s[j]) * (land_s[i] - land_s[j]) > 0)
        total_pairs = n * (n - 1) // 2
        if agree * 2 < total_pairs:
            land_s = [-v for v in land_s]
        cross = [[False] * n for _ in range(n)]
        n_cross = 0
        for i in range(n):
            for j in range(i + 1, n):
                if (comb_s[i] - comb_s[j]) * (land_s[i] - land_s[j]) < 0:
                    cross[i][j] = cross[j][i] = True
                    n_cross += 1
        teeth = [net_teeth_common_layer(pcb_data, nid)
                 for nid in bus.net_ids]
        # Terminal DIVE cost (#622 K=30 fix): per member and candidate
        # layer, each terminal end whose attach-layer set excludes the
        # layer needs one via to transition -- priced at 0.5 crossing
        # units per end (2 vias == 1 crossing, the same equivalence the
        # old teeth term used). This is what the teeth term was FOR,
        # but net_teeth_common_layer reads None on dogbone fanouts
        # (via-terminated stubs have no free tip), so the max-cut
        # priced terminal reality at zero and assigned members to
        # rivers they measurably refused (9-member B river, 8
        # defectors at ~0% acceptance despite the armed 5x tax).
        attach = [net_end_attach_layers(pcb_data, nid)
                  for nid in bus.net_ids]

        # KICAD_BUS_DIVE_SPLIT gates the dive-cost model (default OFF
        # pending the corpus screen): ON prices terminal reality per
        # end; OFF reproduces the legacy teeth-mismatch term exactly
        # (which reads 0 on dogbone/stub-only fanouts -- the K=30
        # blindness -- but is what the K=51 acceptance stack was tuned
        # around: dive-split alone re-membered the rivers and measured
        # 3 failed / 238 vias there vs the legacy 0/202).
        _dive_on = _os.environ.get('KICAD_BUS_DIVE_SPLIT') \
            not in (None, '', '0')

        def _dive(i, layer):
            if _dive_on:
                return 0.5 * sum(1 for ends in attach[i]
                                 if ends and layer not in ends)
            return (1.0 if teeth[i] is not None and teeth[i] != layer
                    else 0.0)

        if _os.environ.get('KICAD_DBG_RIVER_SPLIT'):
            for i, nid in enumerate(bus.net_ids):
                nm = pcb_data.nets[nid].name.split('/')[-1]
                print(f"    [split-dbg] {bus.name} {nm}: tooth "
                      f"{teeth[i]} attach "
                      f"{[sorted(e) for e in attach[i]]}")
        from collections import Counter as _C
        layer_votes = _C(t for t in teeth if t)
        if _os.environ.get('KICAD_BUS_DIVE_SPLIT') not in (None, '',
                                                           '0'):
            for i in range(n):
                for ends in attach[i]:
                    if len(ends) == 1:  # a locked end votes its layer
                        layer_votes[next(iter(ends))] += 1
        # A pour-covered layer cannot host a river: the plane-fragility
        # field overpowers any polite layer nudge and every member
        # defects (measured m2_51: 12/12 B-river members below 50%
        # on-layer, board WORSE than unsplit -- the human's F/B split is
        # cheap only because their board has no pour). Exclude pour
        # layers; when fewer than two free layers remain (2-layer +
        # pour), the group stays whole, disclosed.
        pour_layers = set()
        if not _os.environ.get('KICAD_BUS_RIVER_POUR_OK'):
            # KICAD_BUS_RIVER_POUR_OK=1: deliberate override -- the user
            # accepts a river across the pour (or a probe arm simulates
            # a pour-free board with KICAD_PLANE_FRAGILITY_COST=0).
            for z in (getattr(pcb_data, 'zones', None) or []):
                if getattr(z, 'layer', None):
                    pour_layers.add(z.layer)
                for zl in (getattr(z, 'layers', None) or []):
                    pour_layers.add(zl)
        cands = [l for l, _ in layer_votes.most_common()
                 if l not in pour_layers][:2]
        copper = list(getattr(config, 'layers', None) or [])
        for l in copper:
            if len(cands) >= 2:
                break
            if l not in cands and l not in pour_layers:
                cands.append(l)
        if len(cands) < 2:
            out.append(bus)
            print(f"  Macro rivers: {bus.name} kept whole -- fewer than "
                  f"two pour-free layers "
                  f"(pour on {sorted(pour_layers) or 'none'})")
            continue

        def cost(side):
            mono = sum(1 for i in range(n) for j in range(i + 1, n)
                       if cross[i][j] and side[i] == side[j])
            best = None
            for la, lb in ((cands[0], cands[1]), (cands[1], cands[0])):
                mism = sum(_dive(i, la if side[i] == 0 else lb)
                           for i in range(n))
                if best is None or mism < best[0]:
                    best = (mism, la, lb)
            return mono + best[0], best[1], best[2]

        # whole-group baseline: everyone one river on the majority layer
        base_mism = min(sum(_dive(i, l) for i in range(n))
                        for l in cands)
        base_cost = n_cross + base_mism
        # init: landing-median split (the natural directional partition)
        med = sorted(land_s)[n // 2]
        side = [0 if v < med else 1 for v in land_s]
        cur, la, lb = cost(side)
        improved = True
        while improved:
            improved = False
            for i in range(n):
                side[i] ^= 1
                c2, la2, lb2 = cost(side)
                if c2 < cur and min(side.count(0), side.count(1)) >= 2:
                    cur, la, lb = c2, la2, lb2
                    improved = True
                else:
                    side[i] ^= 1
        if cur >= base_cost or min(side.count(0), side.count(1)) < 2:
            out.append(bus)
            print(f"  Macro rivers: {bus.name} kept whole "
                  f"(split cost {cur} >= whole {base_cost})")
            continue
        for s_val, layer, tag in ((0, la, 'a'), (1, lb, 'b')):
            keep_idx = [i for i in range(n) if side[i] == s_val]
            sub = BusGroup(
                name=f"{bus.name}_r{tag}",
                net_ids=[bus.net_ids[i] for i in keep_idx],
                source_positions=[bus.source_positions[i]
                                  for i in keep_idx],
                target_positions=[bus.target_positions[i]
                                  for i in keep_idx],
                clique_endpoint=bus.clique_endpoint)
            sub.river_layer = layer
            out.append(sub)
            nm = [pcb_data.nets[nid].name.split('/')[-1]
                  for nid in sub.net_ids]
            print(f"  Macro rivers: {sub.name} -> {layer} "
                  f"({len(nm)}: {' '.join(nm)})")
        print(f"  Macro rivers: {bus.name} split cost {cur} vs whole "
              f"{base_cost} (crossing pairs {n_cross})")
        # capacity ceiling disclosure (#622 fix 3): the corridor probe
        # ladder caps at rung 3 = 7 lanes; a bigger river can never be
        # served by one corridor at any rung
        for _sv in (0, 1):
            _cnt = side.count(_sv)
            if _cnt > 7:
                print(f"  Macro rivers: {bus.name} river of {_cnt} "
                      f"exceeds the 7-lane corridor ceiling "
                      f"(probe MAX_RUNG 3) -- capacity shortfall "
                      f"by construction")
    return out


def bus_river_layer_config(cfg_route, config, group, mult=2.0):
    """Per-member layer economics for a macro river (river_layer on the
    group): the assigned layer keeps the group's cheapest cost, every
    other copper layer pays `mult`x -- enough to hold the river on its
    layer, mild enough that terminal surfacing (the landing leg is
    surface copper) and genuine detours stay affordable. The corridor
    probe passes a stiffer mult (its laminar 20x via price otherwise
    pins the probe to the TOOTH's layer and the whole river follows it
    off-plan -- measured m8_8: the B river's corridor probed layers
    [0]). Guarded like the take-1 lane-layer lesson: an explicit user
    layer fence (any cost > 3x the min) is never overridden."""
    layer = getattr(group, 'river_layer', None)
    if not layer:
        return cfg_route
    layers = list(getattr(cfg_route, 'layers', None) or [])
    if layer not in layers:
        return cfg_route
    base = list(cfg_route.layer_costs or []) or [1.0] * len(layers)
    while len(base) < len(layers):
        base.append(1.0)
    pos = [c for c in base if c >= 0]
    if pos and max(pos) > 3.0 * min(pos):
        return cfg_route                 # user fence -- hands off
    lo = min(pos) if pos else 1.0
    shaped = [c if c < 0 else (lo if layers[i] == layer else c * mult)
              for i, c in enumerate(base)]
    from dataclasses import replace as _rep
    return _rep(cfg_route, layer_costs=shaped)


def prepare_victim_bus_context(pcb_data, state, net_id, config,
                               attraction_path, routed_paths,
                               obstacles=None):
    """Full bus stack for a rip victim's reroute -- ONE copy shared by
    the reroute loop and both Phase-3 victim sites: landing-leg
    re-emission (the leg was ripped WITH the result), river layer
    economics at the victim dose (KICAD_BUS_VICTIM_RIVER_MULT, default
    5.0 -- the measured completion arm; no stick, that cascaded), and
    the same terminal-plan refinement the main loop applies (lane
    target / gap leg + waypoints). Returns
    (cfg, attraction_path, leg_segs, group, ball_xy). Non-members pass
    through untouched. Callers must merge leg_segs into the victim's
    RESULT on success (the sync strips unclaimed mid-run copper) and
    retract + re-point the group target at ball_xy on terminal
    failure."""
    group = (getattr(state, 'bus_net_to_group', None) or {}).get(net_id)
    if group is None:
        return config, attraction_path, None, None, None
    from bus_terminal import ensure_landing_leg
    leg_segs = ball = None
    el = ensure_landing_leg(pcb_data, group, net_id, config)
    if el is not None:
        tip, leg_segs, ball = el
        print(f"  Landing leg re-emitted for ripped member "
              f"(tip {tip[0]:.2f},{tip[1]:.2f})")
    import os as _pv_os
    cfg = bus_river_layer_config(
        config, config, group,
        mult=float(_pv_os.environ.get('KICAD_BUS_VICTIM_RIVER_MULT',
                                      '5.0')))
    attraction_path, wps = refine_member_attraction(
        pcb_data, net_id, group, attraction_path,
        getattr(state, 'bus_corridors', None), routed_paths, config,
        obstacles=obstacles)
    if wps:
        from dataclasses import replace as _pv_rep
        cfg = _pv_rep(cfg, corridor_waypoints=list(wps),
                      bus_leg_waypoints=True)
    return cfg, attraction_path, leg_segs, group, ball


def retract_victim_leg(pcb_data, group, net_id, leg_segs, ball_xy):
    """Terminal-failure counterpart of prepare_victim_bus_context: free
    the re-emitted leg and re-point the group target at the BALL so the
    next attempt re-derives from scratch (a restore re-adds the PRE-rip
    result, which carries its own leg)."""
    if not leg_segs:
        return
    from bus_terminal import remove_landing_leg
    n_rm = remove_landing_leg(pcb_data, leg_segs)
    if group is not None and ball_xy is not None \
            and net_id in group.net_ids:
        group.target_positions[group.net_ids.index(net_id)] = ball_xy
    print(f"  Landing leg retracted: {n_rm} seg(s) freed after "
          f"terminal reroute failure")


def _rdp2(pts, eps):
    """2D Ramer-Douglas-Peucker on [(x, y), ...]."""
    if len(pts) < 3:
        return list(pts)
    ax, ay = pts[0]
    bx, by = pts[-1]
    dx, dy = bx - ax, by - ay
    L = math.hypot(dx, dy) or 1e-9
    worst_d, worst_i = -1.0, 0
    for i in range(1, len(pts) - 1):
        px, py = pts[i]
        d = abs((px - ax) * dy - (py - ay) * dx) / L
        if d > worst_d:
            worst_d, worst_i = d, i
    if worst_d <= eps:
        return [pts[0], pts[-1]]
    left = _rdp2(pts[:worst_i + 1], eps)
    return left[:-1] + _rdp2(pts[worst_i:], eps)


def smooth_realized_path(path, eps_cells=2.5, step_cells=1.0):
    """RDP-smooth a realized 3D grid path per LAYER RUN (layer-change
    points are preserved exactly), then densify back to ~step_cells
    spacing so pointwise consumers (terminal_plan_target's per-point
    offsets, the attraction sampler) see the same density as a raw
    path. A guide built from the raw path copies every A* kink and
    post-rip detour into every follower's steering (measured q4_51:
    the worst guide ran 5.97x its straight-line length -- it was
    teaching its member to loop)."""
    if not path or len(path) < 3:
        return path
    runs = []
    cur = [path[0]]
    for p in path[1:]:
        if p[2] != cur[-1][2]:
            runs.append(cur)
            cur = [p]
        else:
            cur.append(p)
    runs.append(cur)
    out = []
    for run in runs:
        simplified = _rdp2([(p[0], p[1]) for p in run], eps_cells)
        lay = run[0][2]
        # OCTILINEAR doglegs (user law: ALL routing stays octilinear):
        # raw RDP segments are arbitrary-angle, which octilinear copper
        # cannot ride exactly -- followers staircase around the guide
        # and pay basin-tax noise. Each simplified segment becomes
        # 45-degree diagonal + axis-aligned remainder.
        octo = [simplified[0]]
        for (x1, y1), (x2, y2) in zip(simplified, simplified[1:]):
            dx, dy = x2 - x1, y2 - y1
            d = min(abs(dx), abs(dy))
            mid = (x1 + d * (1 if dx > 0 else -1 if dx < 0 else 0),
                   y1 + d * (1 if dy > 0 else -1 if dy < 0 else 0))
            if mid != (x1, y1) and mid != (x2, y2):
                octo.append(mid)
            octo.append((x2, y2))
        dense = []
        for (x1, y1), (x2, y2) in zip(octo, octo[1:]):
            L = math.hypot(x2 - x1, y2 - y1)
            n = max(1, int(L / step_cells))
            for i in range(n):
                t = i / n
                dense.append((int(round(x1 + t * (x2 - x1))),
                              int(round(y1 + t * (y2 - y1))), lay))
        dense.append((int(round(octo[-1][0])),
                      int(round(octo[-1][1])), lay))
        out.extend(dense)
    return out


def _build_lane_repair_padcells(pcb_data, config):
    """Cell -> owner-net-ids map of every pad's clearance zone, grid-
    rasterized once per run (pads never move mid-run). Thru-hole pads
    block all layers; SMD only theirs; NPTH has no copper (skipped).
    Margin = track/2 + clearance (the cell is unusable for a track
    center)."""
    import math as _m
    gs = config.grid_step
    margin = config.track_width / 2.0 + config.clearance
    layers = list(getattr(config, 'layers', None) or [])
    all_idx = tuple(range(len(layers)))
    cells = {}
    for fp in pcb_data.footprints.values():
        for pad in fp.pads:
            if pad.pad_type == 'np_thru_hole':
                continue
            if pad.drill and pad.drill > 0:
                lidx = all_idx
            else:
                lidx = tuple(i for i, nm in enumerate(layers)
                             if nm in pad.layers
                             or any(pl.startswith('*') and pl.endswith('.Cu')
                                    for pl in pad.layers))
            if not lidx:
                continue
            hx = pad.size_x / 2.0 + margin
            hy = pad.size_y / 2.0 + margin
            g0x = int(_m.floor((pad.global_x - hx) / gs))
            g1x = int(_m.ceil((pad.global_x + hx) / gs))
            g0y = int(_m.floor((pad.global_y - hy) / gs))
            g1y = int(_m.ceil((pad.global_y + hy) / gs))
            for gx in range(g0x, g1x + 1):
                dx = max(abs(gx * gs - pad.global_x) - pad.size_x / 2.0,
                         0.0)
                for gy in range(g0y, g1y + 1):
                    dy = max(abs(gy * gs - pad.global_y)
                             - pad.size_y / 2.0, 0.0)
                    if _m.hypot(dx, dy) >= margin:
                        continue
                    for li in lidx:
                        cells.setdefault((gx, gy, li),
                                         set()).add(pad.net_id)
    return cells


def repair_lane_feasibility(path, obstacles, config, name='',
                            pcb_data=None, net_id=None):
    """#622 lane-feasibility repair (KICAD_BUS_LANE_REPAIR=1, default
    off): slide statically-blocked attraction-target points to the
    nearest free lateral cell.

    Measured motivation (K=51 pack arm, awx/rip_census.py +
    awx/lane_feas.py, 0826): 202 rip events concentrate in the 2mm bins
    at (124-126, 66-70) -- exactly the bins where planned lane / pack
    targets run through foreign ball pads (SCS0: 37% of its target
    blocked; SDQ3/SDQ5: 3.1mm CONTIGUOUS blocked runs -- and those two
    are the arm's two late-rip failures). The corridor centerline is
    honestly probed, but its OFFSET copies (ribbon rank lanes, the
    follower's +1-pitch pack target) are never checked against copper,
    so they promise positions no search can take; members squeeze into
    the few real cells and rip each other for them. A corridor-shape
    gate cannot see this (bus_1_rb wrap-ratio is only 1.08).

    Repair: for each PAD-blocked interior point, try lateral offsets
    along the local normal out to one ribbon clamp width (4 pitches),
    preferring the previously-chosen side/distance (coherent runs, not
    per-point zigzag). Only points inside a FOREIGN PAD's clearance
    zone are eligible (v2): v1 repaired every blocked point, and at
    K=30 that slid targets off ROUTED SIBLING copper -- but hugging the
    leader's copper is the designed pack behavior (attraction is soft;
    the member packs adjacent), and the win degraded 0/79/0 ->
    0/88/5-rips. Pads never move mid-run, so pad-blockage is exactly
    the "promised position that can never exist" class. Terminal zones
    (~1mm at each end) are exempt: ball-gap lanes are grid-
    quantization-sealed (0824 lore), so exact-planned landing legs read
    blocked in the grid map while being feasible in exact geometry --
    repairing them would shove the landing off its own leg. Points with
    no free cell in reach keep their position (A* pays, as before)."""
    import os as _lr_os
    if _lr_os.environ.get('KICAD_BUS_LANE_REPAIR') in (None, '', '0'):
        return path
    if obstacles is None or not path or len(path) < 8 \
            or pcb_data is None or net_id is None:
        return path
    import math as _m
    padcells = getattr(pcb_data, '_lane_repair_padcells', None)
    if padcells is None:
        padcells = _build_lane_repair_padcells(pcb_data, config)
        pcb_data._lane_repair_padcells = padcells
    gs = config.grid_step
    pitch = max(1, int(round((config.track_width + config.clearance) / gs)))
    max_off = 4 * pitch
    end_skip = max(4, int(round(1.0 / gs)))
    n = len(path)
    out = list(path)
    prev_choice = None  # (side, dist) continuity between adjacent points
    n_blocked = n_elig = n_slid = 0
    max_slide = 0
    for i in range(end_skip, n - end_skip):
        x, y, l = out[i]
        try:
            if not obstacles.is_blocked(x, y, l):
                prev_choice = None
                continue
        except BaseException:
            return path  # map/point disagreement: repair declines whole
        n_blocked += 1
        owners = padcells.get((x, y, l))
        if not owners or not (owners - {net_id}):
            prev_choice = None
            continue  # blocked, but not by a foreign pad: leave it
        n_elig += 1
        a = path[max(0, i - 3)]
        b = path[min(n - 1, i + 3)]
        tx, ty = b[0] - a[0], b[1] - a[1]
        L = _m.hypot(tx, ty)
        if L < 1e-9:
            continue
        nx, ny = -ty / L, tx / L
        cands = []
        if prev_choice is not None:
            cands.append(prev_choice)
        for d in range(1, max_off + 1):
            cands.append((1, d))
            cands.append((-1, d))
        got = None
        for side, d in cands:
            px = int(round(x + nx * side * d))
            py = int(round(y + ny * side * d))
            if px < 0 or py < 0:
                continue
            try:
                free = not obstacles.is_blocked(px, py, l)
            except BaseException:
                continue
            if free:
                got = (side, d, px, py)
                break
        if got is not None:
            side, d, px, py = got
            out[i] = (px, py, l)
            prev_choice = (side, d)
            n_slid += 1
            max_slide = max(max_slide, d)
        else:
            prev_choice = None
    if n_elig:
        print(f"  [lane-repair] {name}: {n_slid}/{n_elig} pad-blocked "
              f"target pt(s) slid (max {max_slide * gs:.2f}mm; "
              f"{n_blocked} blocked total, non-pad left alone)")
    return out


def refine_member_attraction(pcb_data, net_id, group, attraction_path,
                             bus_corridors, routed_paths, config,
                             obstacles=None):
    """FOLLOW_LEADER terminal refinement, ONE copy shared by the main
    loop and the reroute loop: a member following a routed sibling's
    path gets the polarity-aware LANE target (entry-side offset,
    peel-off truncation, landing leg -- terminal_plan_target); a leader
    following its corridor gets the gap-leg refinement
    (terminal_leader_target). Returns (attraction_path, leg_waypoints).

    The reroute loop used to skip this entirely, so every rip victim
    re-aimed at the RAW centerline -- the exact cell stack its routed
    siblings occupy -- and ripped them back (measured p3_51: 85 of 111
    rips originated in the reroute loop; SA0 alone was attacker or
    victim 27 times)."""
    import os as _ra_os
    if attraction_path is None or group is None:
        return attraction_path, None
    nm = (pcb_data.nets[net_id].name if net_id in pcb_data.nets
          else str(net_id))
    if _ra_os.environ.get('KICAD_BUS_FOLLOW_LEADER') in (None, '', '0'):
        return repair_lane_feasibility(attraction_path, obstacles,
                                       config, nm, pcb_data,
                                       net_id), None
    _nb = get_attraction_neighbor(group, net_id, routed_paths or {})
    if _nb is not None and attraction_path is _nb:
        # Fix B (#622): offset a SMOOTHED copy of the leader's realized
        # path, not the raw one -- see smooth_realized_path.
        _nb_s = smooth_realized_path(_nb)
        _tp, _wps = terminal_plan_target(
            group, net_id, _nb_s, config, pcb_data) or (None, None)
        if _tp:
            print(f"  Terminal plan: {nm} lane target ({len(_tp)} pts) "
                  f"from leader path")
            return repair_lane_feasibility(_tp, obstacles, config,
                                           nm, pcb_data, net_id), _wps
        print(f"  Terminal plan: {nm} DEGENERATE -- smoothed leader "
              f"path kept")
        return repair_lane_feasibility(_nb_s, obstacles, config, nm,
                                       pcb_data, net_id), None
    if (bus_corridors and group.name in bus_corridors
            and attraction_path is bus_corridors[group.name]):
        _tl, _wps = terminal_leader_target(
            group, net_id, attraction_path, config, pcb_data) or (None, None)
        if _tl:
            print(f"  Terminal plan: {nm} leader gap leg ({len(_tl)} pts) "
                  f"from corridor")
            return repair_lane_feasibility(_tl, obstacles, config,
                                           nm, pcb_data, net_id), _wps
    return repair_lane_feasibility(attraction_path, obstacles,
                                   config, nm, pcb_data, net_id), None
