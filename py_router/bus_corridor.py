"""Bus corridor planning (#296 R9 / #424): plan ONE centerline per bus group.

The neighbor-attraction bus router has a structural flaw: the guide (middle)
member routes FIRST with no guidance, so the group's corridor is whatever the
guide's solo shortest path happened to be -- chosen with no knowledge that
N-1 siblings must fit beside it. When that corridor threads a one-track gap,
the siblings' attraction loses to real obstacle costs and the group scatters
(measured corpus-wide in #296: 2.57 vs 1.24 vias/member vs the human).

This module plans the corridor BEFORE any member routes, with capacity chosen
SOFTLY (no hard rejection): the group's representative net is probe-routed at
a ladder of inflated clearances -- extra clearance k*(track+clearance) is
"room for k sibling tracks on each side" -- and the winner minimizes

    score = probe_length_mm + (k_max - k) * room_mm * ROOM_WEIGHT

i.e. a corridor with room for the whole bus wins unless it costs too much
length. The winning probe's path (grid coords + layers) is then fed as the
attraction path to EVERY member, including the guide, so the whole group
packs into a corridor that was chosen with group knowledge. Probes commit
nothing; members still route individually with full rip-up, so everything
stays soft -- a member that genuinely cannot follow the corridor routes
around it exactly as before.

The inflation mechanism is the existing wide-power-net machinery: the probe
temporarily gives the representative a track width of track + 2k*(track+
clearance), so the router itself keeps k sibling-rooms of margin per side on
the SHARED obstacle map (track_margin) -- no per-rung map builds. The power
neck-down ladder is disabled for probes (power_tap_neckdown=False) so a wide
probe either routes at its full margin or fails the rung honestly. One probe
map is still built, excluding ALL bus members' stubs (the siblings will pack
into the corridor; their present stubs must not veto it).
"""

from dataclasses import replace

import env_knobs
from typing import Dict, List, Optional, Tuple

from bus_detection import BusGroup, get_bus_routing_order
from obstacle_map import build_base_obstacle_map

# A missing sibling-room costs this many mm of detour before the planner
# prefers the narrower corridor: k rungs below k_max is penalized like
# ROOM_WEIGHT * k * (track + clearance) millimetres of extra length.
ROOM_WEIGHT = 40.0
# Rung cap: k_max = min(ceil((N-1)/2), MAX_RUNG) sibling-rooms per side.
MAX_RUNG = 3
# Probe via-cost multiplier (#296 R9 phase C): the human routes a bus as a
# river on ONE layer; a centerline probed at normal via cost hops layers
# freely and then guides every member incoherently. Pricing vias up in the
# PROBE ONLY makes the planned corridor layer-coherent; members still pay
# normal via costs. Override with KICAD_BUS_CORRIDOR_VIA_MULT (1 = off).
CORRIDOR_VIA_COST_MULT = 4.0


def _sample_dense(path: List[Tuple[int, int, int]], step: int = 1
                  ) -> List[Tuple[int, int, int]]:
    """Densify a simplified (gx, gy, layer) waypoint path so the Rust
    attraction lookup sees a point every `step` grid cells (same contract as
    single_ended_loop's neighbor-path sampling)."""
    if len(path) < 2:
        return list(path)
    out: List[Tuple[int, int, int]] = []
    for (x1, y1, l1), (x2, y2, l2) in zip(path, path[1:]):
        dx, dy = x2 - x1, y2 - y1
        n = max(abs(dx), abs(dy))
        if n == 0:
            out.append((x1, y1, l1))
            continue
        for i in range(0, n, step):
            t = i / n
            out.append((int(round(x1 + dx * t)), int(round(y1 + dy * t)), l1))
    out.append(path[-1])
    return out


def plan_bus_corridors(pcb_data, config, bus_groups: List[BusGroup],
                       verbose: bool = False
                       ) -> Tuple[Dict[str, List[Tuple[int, int, int]]], List[str]]:
    """Plan a corridor centerline per bus group. Returns (corridors,
    demoted_group_names): corridors is {group.name: dense (gx, gy, layer)
    path}; demoted lists groups whose centerline needs more than the
    allowed layer changes even at the laminar probe price -- the caller
    must strip their bus treatment (members route as plain nets). Groups
    whose every probe rung fails are in neither (callers fall back to
    neighbor attraction)."""
    # Deferred import: single_ended_routing must not import this module.
    from single_ended_routing import route_net_with_obstacles
    from net_queries import calculate_route_length

    corridors: Dict[str, List[Tuple[int, int, int]]] = {}
    demoted: List[str] = []
    if not bus_groups:
        return corridors, demoted

    all_member_ids = [nid for g in bus_groups for nid in g.net_ids]
    room_mm = config.track_width + config.clearance
    # #622 honest per-group probe maps (KICAD_BUS_HONEST_PROBE=1): the
    # shared map excludes EVERY member's copper, so a probe is blind to
    # OTHER groups' teeth and plan-time berth legs and can plan its
    # centerline straight through them (the take-1 colander finding:
    # corridors violating their own margin inside foreign balls were
    # exactly the killer groups). Honest mode builds one map per group,
    # excluding only the probed group's own nets -- N_groups cheap
    # builds. Off by default: default --bus behavior unchanged pending
    # the corpus screen.
    import os as _hp_os
    _honest = bool(_hp_os.environ.get('KICAD_BUS_HONEST_PROBE'))
    probe_map = None
    if not _honest:
        probe_map = build_base_obstacle_map(
            pcb_data, config, all_member_ids,
            net_clearances=config.net_clearances)

    print(f"\n=== Bus corridor planning: {len(bus_groups)} group(s), "
          f"room unit {room_mm:.3f}mm"
          f"{' [honest per-group maps]' if _honest else ''} ===")
    for g in bus_groups:
        # #622 part B: a plan-defined river carries its LEARNED corridor
        # (the median member's realized path from the routed board) --
        # known-feasible geometry, so no probe: convert mm -> grid,
        # trim the member-specific terminal teeth/legs off both ends,
        # densify, done. Attraction stays soft downstream exactly as
        # for a probed corridor.
        _pc = getattr(g, '_plan_corridor_mm', None)
        if _pc and len(_pc) >= 2:
            import math as _m_pc
            _gs = config.grid_step
            _lays = list(config.layers or [])
            # trim by the corridor's own ARC: the first/last 1.5mm are
            # the median member's private terminal geometry (its leg /
            # tooth), not the river. (Group source/target positions are
            # NOT a usable reference here -- with exact legs armed they
            # have already been rewritten to leg tips.)
            _arc = [0.0]
            for _a, _b in zip(_pc, _pc[1:]):
                _arc.append(_arc[-1] + _m_pc.hypot(_b[0] - _a[0],
                                                   _b[1] - _a[1]))
            _TRIM = float(_hp_os.environ.get('KICAD_BUS_PLAN_CLIP',
                                             '3.0'))

            def _at_s(s):
                for _i in range(len(_arc) - 1):
                    if s <= _arc[_i + 1] + 1e-9:
                        _sp = _arc[_i + 1] - _arc[_i]
                        _t = 0.0 if _sp < 1e-12 else (s - _arc[_i]) / _sp
                        return (_pc[_i][0]
                                + _t * (_pc[_i + 1][0] - _pc[_i][0]),
                                _pc[_i][1]
                                + _t * (_pc[_i + 1][1] - _pc[_i][1]))
                return (_pc[-1][0], _pc[-1][1])

            if _arc[-1] > 2 * _TRIM + 0.5:
                _s0, _s1 = _TRIM, _arc[-1] - _TRIM
            else:
                _s0, _s1 = 0.0, _arc[-1]
            _clip = [_at_s(_s0) + (_pc[0][2],)]
            for _p, _s in zip(_pc, _arc):
                if _s0 < _s < _s1:
                    _clip.append(_p)
            _clip.append(_at_s(_s1) + (_pc[-1][2],))
            _pc, _lo, _hi = _clip, 0, len(_clip)
            # FLATTEN to one layer + SMOOTH: the median member's raw
            # path carries its own layer hops and lane jitter -- fed
            # raw as the guide, one member's round-1 excursions attract
            # the whole river (measured r4_51: +41 vias over baseline).
            # The corridor rides the river's layer, and the realized
            # centerline is RDP'd octilinear like every other guide.
            _lay_nm = getattr(g, 'river_layer', None)
            if _lay_nm not in _lays:
                from collections import Counter as _Ct
                _lay_nm = _Ct(p[2] for p in _pc).most_common(1)[0][0]
            _li = _lays.index(_lay_nm) if _lay_nm in _lays else 0
            _gp = []
            for x, y, _ln in _pc[_lo:_hi]:
                q = (int(round(x / _gs)), int(round(y / _gs)), _li)
                if not _gp or q != _gp[-1]:
                    _gp.append(q)
            if len(_gp) >= 2:
                from bus_detection import (smooth_realized_path,
                                           prepare_ribbon_plan)
                corridors[g.name] = smooth_realized_path(_gp)
                print(f"  {g.name}: corridor from PLAN "
                      f"({len(_gp)} pts -> {_lay_nm}, "
                      f"{_arc[-1] - _s1 + _s0:.1f}mm terminal arc "
                      f"clipped)")
                prepare_ribbon_plan(pcb_data, config, g,
                                    corridors[g.name])
                continue
        if _honest:
            probe_map = build_base_obstacle_map(
                pcb_data, config, list(g.net_ids),
                net_clearances=config.net_clearances)
        rep = get_bus_routing_order(g)[0]
        rep_name = pcb_data.nets[rep].name if rep in pcb_data.nets else str(rep)
        k_max = min(MAX_RUNG, max(1, (len(g.net_ids)) // 2))
        best = None      # (score, k, length, result)
        # #622 fix A: with plan-time berth legs, probe the TRUNK between
        # STAND-OFF points in free space instead of the rep's real
        # terminals -- the inflated rung width otherwise dies with
        # 'terminal copper would OVERLAP' at the crowded tooth/ball cells
        # (measured: four rivers lost their corridors at K=51 the moment
        # legs moved the targets to the fringe). The spine's job is the
        # highway between the mouths; teeth and legs own the terminals.
        # Gated on the leg records so default --bus probing is untouched.
        _ovr_src = _ovr_tgt = None
        if getattr(g, '_emitted_leg_segs', None) is not None \
                and rep in g.net_ids:
            import math as _m622
            _ki = g.net_ids.index(rep)
            _sx, _sy = g.source_positions[_ki]
            _tx, _ty = g.target_positions[_ki]
            _vx, _vy = _tx - _sx, _ty - _sy
            _L = _m622.hypot(_vx, _vy)
            if _L > 2.0:
                _vx, _vy = _vx / _L, _vy / _L
                _gs = config.grid_step
                _lay_nm = getattr(g, 'river_layer', None)
                _layers = list(config.layers or [])
                _li = (_layers.index(_lay_nm)
                       if _lay_nm in _layers else 0)

                def _row(x, y):
                    gx = int(round(x / _gs))
                    gy = int(round(y / _gs))
                    return (gx, gy, _li, gx * _gs, gy * _gs)

                # A fixed 0.6mm stand-off can land amid the comb teeth
                # (the straight src->tgt line passes THROUGH the comb;
                # measured -0.059mm overlap at rung 0). Walk outward
                # until the point is exact-checked clear at the FATTEST
                # rung's width, so every rung's terminal is legal.
                from bus_terminal import _leg_clear
                _lay_name = (_layers[_li] if _li < len(_layers)
                             else 'F.Cu')
                _w_max = config.track_width + 2 * k_max * room_mm

                def _clear_pt(x0, y0, dx, dy):
                    t = 0.6
                    while t <= 3.01:
                        # snap to grid FIRST; the rounded point is the
                        # one the probe actually starts from.
                        px = round((x0 + dx * t) / _gs) * _gs
                        py = round((y0 + dy * t) / _gs) * _gs
                        if _leg_clear(pcb_data,
                                      [(px, py), (px + 1e-4, py)],
                                      _lay_name, _w_max,
                                      config.clearance, rep):
                            return px, py
                        t += 0.2
                    return None

                _ps = _clear_pt(_sx, _sy, _vx, _vy)
                _pt = _clear_pt(_tx, _ty, -_vx, -_vy)
                if _ps is not None and _pt is not None:
                    _ovr_src = [_row(*_ps)]
                    _ovr_tgt = [_row(*_pt)]
        def _run_ladder(_pm):
            best = None
            for k in range(k_max, -1, -1):
                # Probe via the wide-power mechanism: a track k sibling-rooms
                # wider per side keeps that much margin from every obstacle on
                # the shared map. Neck-down is disabled so the rung fails
                # honestly instead of silently degrading to rung 0.
                # LAMINAR probe: the corridor centerline is the reference every
                # member is attracted to -- if the probe itself wanders layers,
                # "same-layer attraction" chases a different layer every span
                # and no river can form (observed: SDC0's corridor spanned all
                # four layers at the old 4x). The probe pays a much stiffer
                # layer-change price than member routing, so the skeleton stays
                # on one layer unless a change is genuinely unavoidable.
                via_mult = env_knobs.BUS_CORRIDOR_PROBE_VIA_MULT
                cfg_k = replace(
                    config, power_tap_neckdown=False,
                    via_cost=int(round(config.via_cost * via_mult)),
                    power_net_widths={**config.power_net_widths,
                                      rep: config.track_width + 2 * k * room_mm})
                # #622 macro rivers: a river group's corridor must be probed
                # ON its assigned layer -- at the laminar 20x via price the
                # probe pins to the representative's TOOTH layer and the
                # whole river then follows a wrong-layer skeleton (measured
                # m8_8: B river, corridor layers [0]). River groups probe at
                # the NORMAL via price (on-ramp dives are the human's
                # dive-river-dive; mid-path wander is suppressed by the 5x
                # off-river-layer cost instead, and the demotion counter
                # already forgives ramp changes within 3mm of the ends).
                if getattr(g, 'river_layer', None):
                    from bus_detection import bus_river_layer_config
                    cfg_k = bus_river_layer_config(
                        replace(cfg_k, via_cost=config.via_cost),
                        config, g, mult=5.0)
                result = route_net_with_obstacles(
                    pcb_data, rep, cfg_k, _pm,
                    reverse_direction=(g.clique_endpoint == "target"),
                    sources_override=_ovr_src, targets_override=_ovr_tgt)
                if not result or result.get('failed') or not result.get('path'):
                    if verbose:
                        print(f"  {g.name}: rung {k} (+{k * room_mm:.2f}mm "
                              f"clearance) no route")
                    continue
                length = calculate_route_length(
                    result['new_segments'], result.get('new_vias', []), pcb_data)
                score = length + (k_max - k) * room_mm * ROOM_WEIGHT
                if verbose:
                    print(f"  {g.name}: rung {k} routes, length {length:.1f}mm, "
                          f"score {score:.1f}")
                if best is None or score < best[0]:
                    best = (score, k, length, result)
            return best

        best = _run_ladder(probe_map)
        if best is None:
            print(f"  {g.name}: no corridor at any rung -- falling back to "
                  f"neighbor attraction")
            continue
        _score, k, length, result = best
        # Demotion rule: a bus whose CENTERLINE still needs more than
        # KICAD_BUS_MAX_CORRIDOR_LAYER_CHANGES (default 1) layer changes at
        # the stiff probe price has no laminar lane to offer -- routing it
        # "as a bus" would just attract members to a wandering skeleton.
        # Such a group is DEMOTED: no corridor, and the caller strips its
        # bus treatment entirely (members route as plain nets).
        # Count MID-PATH layer changes only: a transition within ~3mm of
        # either endpoint is the river's on-ramp (the dogbone dive from a
        # surface stub into the lane and back out -- the human's exact PCM
        # topology), not wander. Only changes in the middle demote. The
        # path is simplified waypoints, so measure geometric distance.
        import math as _math
        _path = result['path']
        _cum = [0.0]
        for i in range(1, len(_path)):
            _cum.append(_cum[-1] + _math.hypot(
                _path[i][0] - _path[i - 1][0], _path[i][1] - _path[i - 1][1]))
        _total = _cum[-1] or 1.0
        _ramp = 3.0 / config.grid_step
        _changes = sum(
            1 for i in range(1, len(_path))
            if _path[i][2] != _path[i - 1][2]
            and _ramp < _cum[i] < _total - _ramp)
        _max_changes = env_knobs.BUS_MAX_CORRIDOR_LAYER_CHANGES
        if _changes > _max_changes:
            demoted.append(g.name)
            print(f"  {g.name}: corridor needs {_changes} layer change(s) "
                  f"(> {_max_changes}) even at the laminar probe price -- "
                  f"DEMOTED, members route as plain nets")
            continue
        corridors[g.name] = _sample_dense(result['path'], step=1)
        lay = sorted({p[2] for p in result['path']})
        print(f"  {g.name}: corridor via {rep_name} at rung {k} "
              f"(room for {k} sibling(s)/side), {length:.1f}mm, "
              f"{_changes} layer change(s), layers {lay}, "
              f"{len(corridors[g.name])} pts")
        # #622 realism fix 3: DISCLOSE capacity shortfall. The winning
        # rung's own arithmetic bounds the corridor at 2k+1 lanes; a
        # river with more members CANNOT pack it (measured K=30: a
        # 9-member river on a rung-2 corridor -> 0.15 mean acceptance,
        # 8 defectors). Disclosure, not rejection: members still route
        # softly, but the shortfall is data instead of a mystery.
        _lanes = 2 * k + 1
        if len(g.net_ids) > _lanes:
            print(f"  {g.name}: CAPACITY SHORTFALL -- {len(g.net_ids)} "
                  f"members > {_lanes} lane(s) at rung {k}; expect "
                  f"defection/excursions for ~"
                  f"{len(g.net_ids) - _lanes} member(s)")
        from bus_detection import prepare_ribbon_plan
        prepare_ribbon_plan(pcb_data, config, g, corridors[g.name])
    return corridors, demoted
