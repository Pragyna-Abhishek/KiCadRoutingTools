"""Exact-emitted landing legs (#622 take-2, user design).

A bus member landing on a BARE ball inside a ball field cannot thread the
inter-ball gap by search at the routing grid: the raster's cell-granular
stamps close the single legal column (measured: identical copper across a
via-cost 75..400 sweep -- there was no alternative to price -- and the same
pair threads F.Cu with 0 vias at --grid-step 0.05). The fanout solved the
identical problem for ESCAPES with exact-checked lane-runs; this module is
the same idea at the DESTINATION, leg by leg during routing:

  ball -> 45-degree jog into the adjacent inter-column gap -> straight down
  the gap -> out past the field fringe. The leg is EXACT-checked against
  real board geometry (pads, tracks, vias -- DRC-true, no raster) and, when
  clear, emitted as this-net copper BEFORE the member routes. The net then
  has a stub, endpoint derivation targets its free tip at the field edge,
  and the searched route never enters the ball field at all -- like routing
  to a fanout tooth that was created on demand, oriented by the plan.

A leg that fails its exact check emits nothing: the member routes exactly
as before (the dive-up stays as fallback). Emitted legs are recorded on
pcb_data._emitted_legs for the output writer.
"""

import math
from typing import Dict, List, Optional, Tuple

from kicad_parser import PCBData, Segment


def _pt_seg_d(px, py, x1, y1, x2, y2):
    dx, dy = x2 - x1, y2 - y1
    L2 = dx * dx + dy * dy
    t = 0.0 if L2 <= 0 else max(0.0, min(1.0, ((px - x1) * dx
                                               + (py - y1) * dy) / L2))
    return math.hypot(px - (x1 + t * dx), py - (y1 + t * dy))


def _seg_seg_d(x1, y1, x2, y2, u1, v1, u2, v2):
    """Segment-segment distance, 0 when they properly intersect.
    Endpoint distances alone are BLIND to a crossing (two legs crossed
    mid-span with every endpoint 0.5mm clear -- shipped an F.Cu short,
    h1_8 SA10xSDQ5)."""
    d1x, d1y = x2 - x1, y2 - y1
    d2x, d2y = u2 - u1, v2 - v1
    denom = d1x * d2y - d1y * d2x
    if abs(denom) > 1e-12:
        t = ((u1 - x1) * d2y - (v1 - y1) * d2x) / denom
        s = ((u1 - x1) * d1y - (v1 - y1) * d1x) / denom
        if 0.0 <= t <= 1.0 and 0.0 <= s <= 1.0:
            return 0.0
    return min(_pt_seg_d(u1, v1, x1, y1, x2, y2),
               _pt_seg_d(u2, v2, x1, y1, x2, y2),
               _pt_seg_d(x1, y1, u1, v1, u2, v2),
               _pt_seg_d(x2, y2, u1, v1, u2, v2))


def _leg_clear(pcb_data: PCBData, pts: List[Tuple[float, float]],
               layer: str, width: float, clearance: float,
               net_id: int) -> bool:
    """Exact clearance check of the leg polyline against real geometry --
    the fanout _seg_conflict pattern, standalone over pcb_data. Foreign
    pads (their true shape radius for circles, circumscribed for rects),
    foreign same-layer segments, and all via barrels are checked; own-net
    copper and pads are exempt (the leg lands ON the own ball)."""
    hw = width / 2.0
    for (x1, y1), (x2, y2) in zip(pts, pts[1:]):
        for fp in pcb_data.footprints.values():
            for p in fp.pads:
                if p.net_id == net_id:
                    continue
                if getattr(p, 'drill', 0):
                    r = max(p.size_x, p.size_y) / 2.0
                elif layer in p.layers:
                    if p.shape == 'circle':
                        r = p.size_x / 2.0
                    else:
                        r = math.hypot(p.size_x, p.size_y) / 2.0
                else:
                    continue
                if _pt_seg_d(p.global_x, p.global_y, x1, y1, x2, y2) \
                        < r + hw + clearance - 1e-6:
                    return False
        for s in pcb_data.segments:
            if s.net_id == net_id or s.layer != layer:
                continue
            if _seg_seg_d(x1, y1, x2, y2, s.start_x, s.start_y,
                          s.end_x, s.end_y) \
                    < s.width / 2.0 + hw + clearance - 1e-6:
                return False
        for v in pcb_data.vias:
            if v.net_id == net_id:
                continue
            if _pt_seg_d(v.x, v.y, x1, y1, x2, y2) \
                    < v.size / 2.0 + hw + clearance - 1e-6:
                return False
    return True


def remove_landing_leg(pcb_data: PCBData, leg_segs: List[Segment]) -> int:
    """Retract an emitted landing leg after its member terminally failed
    to route: the leg is copper for a route that never arrived -- left in
    place it ships as an attached spur (and re-poisons the next attempt's
    endpoint derivation). Removes the segments from pcb_data.segments and
    the _emitted_legs ledger by IDENTITY; returns the count removed."""
    if not leg_segs:
        return 0
    ids = {id(s) for s in leg_segs}
    pcb_data.segments[:] = [s for s in pcb_data.segments
                            if id(s) not in ids]
    legs = getattr(pcb_data, '_emitted_legs', None)
    if legs:
        legs[:] = [s for s in legs if id(s) not in ids]
    return len(ids)


def _ball_context(pcb_data: PCBData, ball_xy: Tuple[float, float]):
    """Owning footprint + local pitch + field pads for a bare ball, or
    None when the ball is not in a modelable field. Extracted from
    emit_landing_leg (the 3mm-neighborhood/whole-footprint lesson, f1_2)."""
    bx, by = ball_xy
    near = []
    owner = None
    for fp in pcb_data.footprints.values():
        for p in fp.pads:
            d = math.hypot(p.global_x - bx, p.global_y - by)
            if d < 3.0:
                near.append((d, p))
                if d < 0.2:
                    owner = fp
    near.sort(key=lambda t: t[0])
    if owner is None or len(near) < 3:
        return None
    pitch = math.hypot(near[1][1].global_x - near[0][1].global_x,
                       near[1][1].global_y - near[0][1].global_y)
    if pitch < 0.3:
        return None
    return owner, pitch, list(owner.pads)


def _leg_candidates(ctx, ball_xy: Tuple[float, float],
                    approach_from: Tuple[float, float],
                    lane_shifts: int = 1, lane_pitch: float = 0.4,
                    grid: float = 0.1):
    """Candidate landing legs for one ball, sorted approach-compatible
    first then shortest run: (compat_rank, run, pts, tip) tuples, NOT yet
    exact-checked. lane_shifts>1 adds lane-shifted variants of each gap
    line (k*lane_pitch further out) -- capacity in the wide inter-block
    channels for the berth planner; the jog stretches to stay 45-degree."""
    owner, pitch, field = ctx
    bx, by = ball_xy
    ax, ay = approach_from[0] - bx, approach_from[1] - by

    def _leg_for(side, lsgn, shift):
        exit_x_axis = side in ('e', 'w')
        sgn = 1.0 if side in ('e', 's') else -1.0
        off = pitch / 2.0 + shift * lane_pitch
        if exit_x_axis:
            gap_line = by + lsgn * off
        else:
            gap_line = bx + lsgn * off
        # The tip must clear the WHOLE field on the exit axis, not just
        # the lane's own blocking pads: an empty lane (the wide
        # inter-block channel) used to stop the walk at the ball and
        # plant the tip INSIDE the field -- the trunk then traveled the
        # channel to reach it, squatting siblings' planned lanes
        # (measured g3_8: 4 of 8 berth legs blocked at emission).
        extent = bx if exit_x_axis else by
        for p in field:
            u = p.global_x if exit_x_axis else p.global_y
            if (u - extent) * sgn > 0:
                extent = u
        tip_u = extent + sgn * (pitch / 2.0 + 0.2)
        if exit_x_axis:
            jog = (bx + sgn * off, gap_line)
            tip = (tip_u, gap_line)
        else:
            jog = (gap_line, by + sgn * off)
            tip = (gap_line, tip_u)
        pts = [(bx, by), jog, tip]
        # Octilinear grid on-ramp (user law: ALL routing octilinear):
        # gap lines are ball-pitch geometry, off the routing grid, so a
        # route attaching at a raw tip needs a tiny any-angle connector
        # (measured: 29um x 39um segment at SDQ7's berth). Extend the
        # leg with a 45-degree piece to the nearest grid line and an
        # axis piece to a GRID-ALIGNED free tip -- routes then land on
        # the tip exactly.
        g = grid
        if exit_x_axis:
            gy = round(gap_line / g) * g
            d_lat = abs(gy - gap_line)
            if d_lat > 1e-9:
                p45 = (tip_u + sgn * d_lat, gy)
                pts.append(p45)
            else:
                p45 = tip
            gx = (math.floor(p45[0] / g) if sgn < 0
                  else math.ceil(p45[0] / g)) * g
            if abs(gx - p45[0]) > 1e-9:
                pts.append((gx, gy))
            tip = (round(gx, 6), round(gy, 6))
        else:
            gx = round(gap_line / g) * g
            d_lat = abs(gx - gap_line)
            if d_lat > 1e-9:
                p45 = (gx, tip_u + sgn * d_lat)
                pts.append(p45)
            else:
                p45 = tip
            gy = (math.floor(p45[1] / g) if sgn < 0
                  else math.ceil(p45[1] / g)) * g
            if abs(gy - p45[1]) > 1e-9:
                pts.append((gx, gy))
            tip = (round(gx, 6), round(gy, 6))
        pts[-1] = tip
        run = abs(tip_u - (bx if exit_x_axis else by))
        return run, pts, tip

    # Side order: PACK-FACING first (the tip must face the arriving pack
    # -- nearest-boundary alone sent legs out the far side and the route
    # had to wrap the band, 1 failed, measured f2_2), then perpendicular,
    # then away-facing; shortest run within a rank. Per side, the gap
    # lateral toward the approach first. NOTE the sign convention paid
    # for twice: svec is the EXIT direction and (ax,ay) points from the
    # ball TO the pack, so exit.dot(approach) > 0 is the GOOD side (the
    # original code's `compat` flag named it backwards and relied on the
    # ascending sort; a faithful-looking refactor of it inverted every
    # side order -- measured g4_8, all-east/north berths).
    _svec = {'e': (1.0, 0.0), 'w': (-1.0, 0.0),
             's': (0.0, 1.0), 'n': (0.0, -1.0)}
    _al = math.hypot(ax, ay) or 1.0
    cands = []
    for side in ('e', 'w', 's', 'n'):
        if side in ('e', 'w'):
            pref = 1.0 if ay >= 0 else -1.0
        else:
            pref = 1.0 if ax >= 0 else -1.0
        dot_u = (_svec[side][0] * ax + _svec[side][1] * ay) / _al
        rank = 0 if dot_u > 0.3 else (1 if dot_u > -0.5 else 2)
        for lsgn in (pref, -pref):
            for shift in range(lane_shifts):
                c = _leg_for(side, lsgn, shift)
                if c is not None:
                    cands.append((rank,) + c)
    cands.sort(key=lambda t: (t[0], t[1]))
    return cands


def _polyline_min_dist(a: List[Tuple[float, float]],
                       b: List[Tuple[float, float]]) -> float:
    """Min distance between two polylines, 0 when any pair of their
    segments crosses (see _seg_seg_d)."""
    best = float('inf')
    for (x1, y1), (x2, y2) in zip(a, a[1:]):
        for (u1, v1), (u2, v2) in zip(b, b[1:]):
            best = min(best, _seg_seg_d(x1, y1, x2, y2, u1, v1, u2, v2))
            if best <= 0.0:
                return 0.0
    return best


def plan_landing_berths(pcb_data: PCBData, group, config,
                        bare_net_ids: List[int]) -> Dict[int, Optional[list]]:
    """#622 take-2 berth assignment (user design, item 2): plan every
    bare-ball member's landing gap UP FRONT with a non-blocking solve, so
    gap blocking is impossible by construction. First-come emission
    measured at K=8: early members took the near-side gaps, later legs
    failed their exact checks and fell through to far-side exits (tips
    past the field at x=146.5) forcing full-field orbits (ratio 3.7,
    20 vias/8 nets).

    Candidates per member: approach-compatible short legs (near-edge rule
    for perpendicular exits: only toward the member's OWN block fringe),
    lane-shifted channel variants for capacity, exact-checked against the
    BASE board (no first-come bias). Solve: maximize members berthed,
    then minimize total run (DFS for small groups, greedy above 12).
    A member with no conflict-free berth gets None = NO leg: an organized
    dive beats a pathological surface exit (the human's own overflow
    policy -- 29 dives, median 0.79mm from the ball)."""
    n_ids = list(bare_net_ids)
    per_net = {}
    fld_pts = []
    for nid in n_ids:
        k = group.net_ids.index(nid)
        ball = group.target_positions[k]
        ctx = _ball_context(pcb_data, ball)
        if ctx is None:
            per_net[nid] = []
            continue
        _n = float(len(group.net_ids))
        appr = (sum(p[0] for p in group.source_positions) / _n,
                sum(p[1] for p in group.source_positions) / _n)
        _, pitch, field = ctx
        if not fld_pts:
            fld_pts = [(p.global_x, p.global_y) for p in field]
        bx, by = ball
        # near-edge rule for perpendicular exits: the ball's own block
        # fringe only (a south exit from a north-block ball crosses the
        # whole field -- measured K=8, SDQ0 tip 6.2mm through).
        f_ys = sorted(p.global_y for p in field)
        _al = math.hypot(appr[0] - bx, appr[1] - by) or 1.0
        _au = ((appr[0] - bx) / _al, (appr[1] - by) / _al)
        cands = []
        for side_rank, run, pts, tip in _leg_candidates(
                ctx, ball, appr, lane_shifts=3,
                grid=getattr(config, 'grid_step', 0.1)):
            if side_rank >= 2:
                continue  # never berth out the far side
            ex, ey = pts[-1][0] - pts[-2][0], pts[-1][1] - pts[-2][1]
            _el2 = math.hypot(ex, ey) or 1.0
            # exits ON the approach axis outrank perpendicular ones:
            # run-length alone chose 1.4mm north pokes over the 4.6mm
            # west-fringe legs the accepted K=4 used (measured g3_8) --
            # the pack arrives on the axis, and a perpendicular tip
            # makes it wrap a corner.
            align = -(ex * _au[0] + ey * _au[1]) / _el2  # 0 best
            if abs(ex) < 1e-6:
                # n/s exit (vertical last segment): the ball's OWN block
                # fringe only -- a cross-field exit is a wrap by
                # construction. e/w exits need no such rule: compat
                # already pins them to the approach side.
                mid_y = (f_ys[0] + f_ys[-1]) / 2.0
                if (by <= mid_y) != (tip[1] <= mid_y):
                    continue
            if not _leg_clear(pcb_data, pts, config.layers[0],
                              config.track_width, config.clearance, nid):
                continue
            # solve cost: run in mm plus an axis penalty -- perpendicular
            # exits pay +4mm-equiv, so they win only when the axis side
            # is genuinely unavailable, count still dominating both.
            cost = run + (1.0 + align) * 4.0
            cands.append((cost, pts, tip))
        cands.sort(key=lambda c: c[0])
        per_net[nid] = cands

    # plan-time conflict floor carries +0.1 headroom over the DRC
    # minimum: lanes planned at the exact floor left no room for
    # the trunk's attach wiggle at the fringe mouth (SDQ5 blocked
    # at emission by its neighbor's arrival, g5_8).
    min_gap = config.track_width + config.clearance + 0.05

    def _conflict(pa, pb):
        return _polyline_min_dist(pa, pb) < min_gap - 1e-6

    # Tip ORDER term: an inversion between the berth order along the
    # field perimeter and the group's PHYSICAL order is a crossing the
    # pack must pay somewhere (2 vias) -- measured g6_8: 8 inversions,
    # 16 vias, exactly 2 per inversion. Perimeter coordinate s walks
    # north fringe -> approach-side fringe -> south fringe, so lanes
    # wrapping a corner still order correctly against the west lanes.
    # Fringe classification must be against the FIELD hull, not the
    # group's own ball bbox: a west tip whose gap line rides above the
    # group's topmost ball is still a WEST tip (measured g8_8 -- it was
    # scored as a wrapped north lane, inverting every order decision
    # around it). fld_* comes from the owner footprint's pads.
    fx_min = min((x for x, y in fld_pts), default=0.0)
    fy_min = min((y for x, y in fld_pts), default=0.0)
    fy_max = max((y for x, y in fld_pts), default=0.0)

    def _perim_s(tip):
        tx, ty = tip
        if tx < fx_min - 0.05:        # west fringe: order by y
            return ty
        if ty < fy_min - 0.05:        # north fringe: deeper east = more north
            return fy_min - (tx - fx_min)
        if ty > fy_max + 0.05:        # south fringe: deeper east = more south
            return fy_max + (tx - fx_min)
        return ty
    INV_COST = 4.0  # mm-equiv per crossing (2 vias + detour)

    # DFS in the group's PHYSICAL order so the inversion count against
    # already-placed members accumulates monotonically (prunable).
    # Pairwise conflicts and perimeter coords are precomputed -- the
    # inner loop touches no geometry.
    order = [nid for nid in group.net_ids if nid in per_net]
    flat = []                 # (nid, run, pts, tip, s), global index
    slots = {}                # nid -> [global candidate indices]
    for nid in order:
        slots[nid] = []
        for run, pts, tip in per_net[nid]:
            slots[nid].append(len(flat))
            flat.append((nid, run, pts, tip, _perim_s(tip)))
    n_c = len(flat)
    conf = [set() for _ in range(n_c)]
    for a in range(n_c):
        for b in range(a + 1, n_c):
            if flat[a][0] != flat[b][0] and \
                    _conflict(flat[a][2], flat[b][2]):
                conf[a].add(b)
                conf[b].add(a)
    # Net-level conflict graph -> connected components, each solved by
    # its OWN exact DFS: members whose candidates never conflict live in
    # different ball blocks and don't belong in one search space -- the
    # component sizes are what keep K=15..51 groups exact instead of
    # falling back to a greedy that ignores the order term. Inversions
    # are counted WITHIN a component; cross-component pairs sit on
    # geometry where the order is forced anyway.
    net_conf = {nid: set() for nid in order}
    for a in range(n_c):
        for b in conf[a]:
            net_conf[flat[a][0]].add(flat[b][0])
    comps = []
    seen_c = set()
    for nid in order:
        if nid in seen_c:
            continue
        stack, comp = [nid], []
        while stack:
            cur = stack.pop()
            if cur in seen_c:
                continue
            seen_c.add(cur)
            comp.append(cur)
            stack.extend(n for n in net_conf[cur] if n not in seen_c)
        comps.append([n for n in order if n in comp])

    import os as _dbg_os
    if _dbg_os.environ.get('KICAD_DBG_BERTHS'):
        for nid in order:
            nm = pcb_data.nets[nid].name.split('/')[-1]
            k = group.net_ids.index(nid)
            _b = group.target_positions[k]
            print(f"    [berth-dbg] {nm} ball=({_b[0]:.2f},{_b[1]:.2f}): "
                  + " | ".join(
                      f"({flat[ci][3][0]:.2f},{flat[ci][3][1]:.2f})"
                      f" c={flat[ci][1]:.2f} s={flat[ci][4]:.2f}"
                      for ci in slots[nid]))

    chosen = {}
    for order_c in comps:
        best = {'count': -1, 'cost': float('inf'), 'picks': {}}

        def _dfs(i, picks, taken, cost, placed):
            if i == len(order_c):
                if (placed, -cost) > (best['count'], -best['cost']):
                    best.update(count=placed, cost=cost, picks=dict(picks))
                return
            rem = len(order_c) - i
            if placed + rem < best['count'] or \
                    (placed + rem == best['count'] and cost >= best['cost']):
                return
            nid = order_c[i]
            for ci in slots[nid]:
                if taken & conf[ci]:
                    continue
                s = flat[ci][4]
                inv = sum(1 for pj in picks.values()
                          if pj is not None and flat[pj][4] > s)
                picks[nid] = ci
                taken.add(ci)
                _dfs(i + 1, picks, taken,
                     cost + flat[ci][1] + INV_COST * inv, placed + 1)
                taken.discard(ci)
                del picks[nid]
            picks[nid] = None
            _dfs(i + 1, picks, taken, cost, placed)
            del picks[nid]

        if len(order_c) <= 12:
            _dfs(0, {}, set(), 0.0, 0)
            for nid, ci in best['picks'].items():
                chosen[nid] = None if ci is None \
                    else (flat[ci][2], flat[ci][3])
            if _dbg_os.environ.get('KICAD_DBG_BERTHS'):
                print(f"    [berth-dbg] component {len(order_c)} nets: "
                      f"count={best['count']} cost={best['cost']:.2f}")
        else:
            # component too large for exact search: greedy in candidate-
            # scarcity order, order term dropped (disclosed).
            print(f"  Berth plan: component of {len(order_c)} nets "
                  f"exceeds exact-solve cap; greedy, no order term")
            for nid in sorted(order_c, key=lambda nid: len(per_net[nid])):
                pick = None
                for run, pts, tip in per_net[nid]:
                    if not any(v is not None and _conflict(pts, v[0])
                               for v in chosen.values()):
                        pick = (pts, tip)
                        break
                chosen[nid] = pick
    plan = {}
    for nid in n_ids:
        v = chosen.get(nid)
        plan[nid] = list(v[0]) if v is not None else None
        nm = pcb_data.nets[nid].name if nid in pcb_data.nets else str(nid)
        if v is None:
            print(f"  Berth plan: {nm} -> DIVE (no conflict-free berth)")
        else:
            print(f"  Berth plan: {nm} -> tip ({v[1][0]:.2f},{v[1][1]:.2f})")
    # Alternates for emission-time fallback: a planned leg can be blocked
    # by earlier siblings' TRUNK copper the plan cannot see. The member
    # may then use another of its own measured candidates, but only one
    # that conflicts with NO sibling's planned berth -- the plan stays
    # authoritative for everyone else (a free-for-all fallback would
    # resurrect the first-come contention the plan exists to remove).
    plan['_alternates'] = {
        nid: [pts for _c, pts, _t in per_net[nid]
              if not any(o != nid and plan.get(o) is not None
                         and _conflict(pts, plan[o])
                         for o in n_ids)]
        for nid in n_ids}
    return plan


def ensure_landing_leg(pcb_data: PCBData, group, net_id: int, config):
    """One-stop landing-leg emission for a bus member: bareness test,
    lazy berth-plan solve (cached on the group), planned-leg/alternate
    emission, and the group-target update to the new tip. Returns
    (tip, [Segment...], ball_xy) or None (not armed, not bare, planned
    DIVE, or blocked). SHARED by the main loop and the reroute loop --
    a ripped member's leg is ripped with its result, and re-entering
    without re-emission meant every rip demoted the member to a
    dive-at-the-ball loose net.

    With plan-time emission (emit_group_berth_legs), the group carries
    _emitted_leg_segs: a still-live record returns idempotently (the
    caller merges the SAME segs into this route's result, which is what
    keeps the sync from stripping them); a record whose segments were
    ripped away re-emits and refreshes the record."""
    import os as _os
    if not _os.environ.get('KICAD_BUS_EXACT_LEGS'):
        return None
    if group is None or net_id not in group.net_ids:
        return None
    recs = getattr(group, '_emitted_leg_segs', None)
    _true_ball = None
    if recs is not None and net_id in recs:
        _tip, _segs, _ball = recs[net_id]
        _seg_ids = {id(s) for s in pcb_data.segments}
        if all(id(s) in _seg_ids for s in _segs):
            return recs[net_id]          # legs still live: idempotent
        del recs[net_id]                 # ripped away: fall through, re-emit
        _true_ball = _ball               # target_positions still holds the
        # OLD TIP after a rip (only terminal retraction re-points it) --
        # emitting with the tip as the "ball" plants a leg in space.
    k = group.net_ids.index(net_id)
    tgt = _true_ball if _true_ball is not None \
        else group.target_positions[k]

    def _bare(nid, t):
        return not any(
            s.net_id == nid
            and min(abs(s.start_x - t[0]) + abs(s.start_y - t[1]),
                    abs(s.end_x - t[0]) + abs(s.end_y - t[1])) < 1.0
            for s in pcb_data.segments)

    if not _bare(net_id, tgt):
        return None
    _n = float(len(group.net_ids))
    appr = (sum(p[0] for p in group.source_positions) / _n,
            sum(p[1] for p in group.source_positions) / _n)
    planned = None
    has_plan = False
    alts = None
    if _os.environ.get('KICAD_BUS_BERTHS'):
        plan = getattr(group, '_berth_plan', None)
        if plan is None:
            bare_ids = [m for j, m in enumerate(group.net_ids)
                        if _bare(m, group.target_positions[j])]
            plan = plan_landing_berths(pcb_data, group, config, bare_ids)
            setattr(group, '_berth_plan', plan)
        if net_id in plan:
            planned = plan[net_id]
            has_plan = True
            alts = (plan.get('_alternates') or {}).get(net_id)
    if has_plan and planned is None:
        return None                      # planned DIVE: emit nothing
    res = emit_landing_leg(pcb_data, net_id, tgt, appr, config,
                           planned_pts=planned, alternate_pts=alts)
    if res is None:
        return None
    tip, segs = res
    group.target_positions[k] = tip
    out = (tip, segs, tgt)
    if recs is not None:
        recs[net_id] = out
    return out


def yield_blocking_legs(pcb_data: PCBData, state, blocked_cells,
                        grid_step: float, bus_net_to_group,
                        routed_results, exclude_net_id: int,
                        radius_cells: float = 8.0, config=None):
    """#622 leg-yield: plan-time berth legs are THIS RUN's reservations,
    not foreign pre-existing copper -- but until their member routes
    they carry no result, so the rip attribution classified them as
    committed-by-an-earlier-step and the ladder surrendered ('no
    rippable blockers' -> boxed_in_static on nets walled by their own
    siblings' reservations; measured t8_51: SDQ6 walled by unrouted
    SCS0's leg while SCS0 stranded too).

    Given the failed search's blocked frontier cells, retract every
    live leg belonging to an UNROUTED in-scope member whose polyline
    passes within radius_cells of a blocked cell (frontier cells sit a
    clearance margin INSIDE the wall, so the radius must cover
    width+clearance stamps -- measured 5.7 cells to the load-bearing
    leg at a 138-cell sealed ball pocket): remove the copper,
    clear the group record (the member re-emits at its own route time,
    alternates/dive as fallback), re-point the group target at the
    ball, and scrub the net's obstacle-cache/working-map stamps.
    Returns the yielded net names (empty list = nothing to yield)."""
    import os as _ly_os
    _dbg = bool(_ly_os.environ.get('KICAD_DBG_LEGYIELD'))
    if not blocked_cells:
        if _dbg:
            print("    [legyield-dbg] no blocked cells")
        return []
    legs_by_net = {}
    for s in (getattr(pcb_data, '_emitted_legs', None) or []):
        legs_by_net.setdefault(s.net_id, []).append(s)
    if not legs_by_net:
        if _dbg:
            print("    [legyield-dbg] no emitted legs ledger")
        return []
    live_ids = {id(s) for s in pcb_data.segments}
    cells = [(c[0], c[1]) for c in blocked_cells]
    yielded = []
    for bid, segs in legs_by_net.items():
        nm_d = (pcb_data.nets[bid].name.split('/')[-1]
                if bid in pcb_data.nets else str(bid))
        if bid == exclude_net_id or bid in (routed_results or {}):
            if _dbg:
                print(f"    [legyield-dbg] {nm_d}: skip "
                      f"({'self' if bid == exclude_net_id else 'routed'})")
            continue
        group = (bus_net_to_group or {}).get(bid)
        if group is None:
            if _dbg:
                print(f"    [legyield-dbg] {nm_d}: skip (no group)")
            continue
        segs = [s for s in segs if id(s) in live_ids]
        if not segs:
            if _dbg:
                print(f"    [legyield-dbg] {nm_d}: skip (legs not live)")
            continue
        near = False
        best_d = 1e9
        for s in segs:
            x1, y1 = s.start_x / grid_step, s.start_y / grid_step
            x2, y2 = s.end_x / grid_step, s.end_y / grid_step
            for (cx, cy) in cells:
                d = _pt_seg_d(cx, cy, x1, y1, x2, y2)
                best_d = min(best_d, d)
                if d <= radius_cells:
                    near = True
                    break
            if near:
                break
        if not near:
            if _dbg:
                print(f"    [legyield-dbg] {nm_d}: skip (nearest "
                      f"blocked cell {best_d:.1f} cells)")
            continue
        remove_landing_leg(pcb_data, segs)
        _tip = _ball = None
        recs = getattr(group, '_emitted_leg_segs', None)
        if recs and bid in recs:
            _tip, _s2, _ball = recs.pop(bid)
            if bid in group.net_ids:
                group.target_positions[group.net_ids.index(bid)] = _ball
        # refresh the yielded net's obstacle stamps IN PLACE: remove the
        # stale entry (it still stamps the dead leg) and immediately
        # recompute from post-yield pcb_data -- deleting without
        # recomputing left the net's TEETH unstamped too, and a later
        # net could route through them (the copper-only-undo short,
        # w1_51, was the same disease on the undo side).
        try:
            from obstacle_cache import (remove_net_obstacles_from_cache,
                                        add_net_obstacles_from_cache,
                                        precompute_net_obstacles)
            if getattr(state, 'working_obstacles', None) is not None \
                    and getattr(state, 'net_obstacles_cache', None) \
                    is not None and bid in state.net_obstacles_cache:
                remove_net_obstacles_from_cache(
                    state.working_obstacles,
                    state.net_obstacles_cache[bid])
                state.net_obstacles_cache[bid] = \
                    precompute_net_obstacles(pcb_data, bid, config)
                add_net_obstacles_from_cache(
                    state.working_obstacles,
                    state.net_obstacles_cache[bid])
        except Exception:
            pass
        nm = (pcb_data.nets[bid].name if bid in pcb_data.nets
              else str(bid))
        yielded.append((nm, bid, group, segs, _tip, _ball))
        print(f"  Leg yield: {nm}'s berth leg retracted (unclaimed "
              f"reservation walling the frontier); it re-emits at its "
              f"own route time")
    return yielded


def unyield_legs(pcb_data: PCBData, yielded, state=None,
                 config=None) -> int:
    """Transactional undo for yield_blocking_legs: the retry FAILED, so
    the retraction bought nothing -- put the exact same leg segments
    back (board unchanged since the yield: pure undo, no re-check),
    restore the group record and the tip target, AND re-stamp the net's
    obstacle-cache/working-map entry. The copper-only undo shipped a
    SHORT: the yield had scrubbed the leg's stamps from the working
    map, so a later net routed straight through the restored leg
    (measured w1_51: SDQ9 crossing SDQ12's leg on F.Cu). Without the
    undo at all, a fruitless yield perturbs the whole rest of the run
    (v3_51: +22 vias +25 rips)."""
    n = 0
    for (_nm, bid, group, segs, tip, ball) in yielded:
        pcb_data.segments.extend(segs)
        ledger = getattr(pcb_data, '_emitted_legs', None)
        if ledger is None:
            ledger = []
            setattr(pcb_data, '_emitted_legs', ledger)
        ledger.extend(segs)
        recs = getattr(group, '_emitted_leg_segs', None)
        if recs is not None and tip is not None:
            recs[bid] = (tip, segs, ball)
        if tip is not None and bid in group.net_ids:
            group.target_positions[group.net_ids.index(bid)] = tip
        if state is not None and config is not None:
            try:
                from obstacle_cache import (precompute_net_obstacles,
                                            add_net_obstacles_from_cache)
                if getattr(state, 'net_obstacles_cache', None)                         is not None:
                    state.net_obstacles_cache[bid] =                         precompute_net_obstacles(pcb_data, bid, config)
                    if getattr(state, 'working_obstacles', None)                             is not None:
                        add_net_obstacles_from_cache(
                            state.working_obstacles,
                            state.net_obstacles_cache[bid])
            except Exception:
                pass
        n += 1
    return n


def emit_group_berth_legs(pcb_data: PCBData, group, config) -> list:
    """Fix A (#622): emit EVERY planned berth leg at PLAN TIME, on the
    base board, before corridors are probed and before any member
    routes. Three things follow by construction: berths are reserved
    REAL copper (soft plans could be squatted by sibling trunks -- the
    blocked-at-emission failure mode); every endpoint derivation --
    corridor probes, member routes, attraction terminal plans -- targets
    the leg TIP at the field fringe instead of the raw ball deep in the
    field (measured q4_51: all six spines and 30 of 41 guides ended
    in-field, dragging followers across the berth zone); and the probe
    terminal lands in free space instead of a crowded ball/tooth cell
    (the 'terminal copper would OVERLAP' rung rejections). Returns the
    net ids whose legs were emitted (the caller MUST refresh those nets'
    obstacle-cache entries: the base map and per-net cache are built
    BEFORE plan-time emission, so without a refresh the legs are
    INVISIBLE to the router -- measured w5_51: two A* commits crossed
    live legs whose stamps never existed). Idempotent per group."""
    import os as _os
    if not (_os.environ.get('KICAD_BUS_EXACT_LEGS')
            and _os.environ.get('KICAD_BUS_BERTHS')):
        return []
    if getattr(group, '_emitted_leg_segs', None) is not None:
        return []
    setattr(group, '_emitted_leg_segs', {})
    emitted = []
    for nid in list(group.net_ids):
        if ensure_landing_leg(pcb_data, group, nid, config) is not None:
            emitted.append(nid)
    return emitted


def emit_landing_leg(pcb_data: PCBData, net_id: int,
                     ball_xy: Tuple[float, float],
                     approach_from: Tuple[float, float],
                     config,
                     planned_pts: Optional[list] = None,
                     alternate_pts: Optional[List[list]] = None
                     ) -> Optional[Tuple[float, float]]:
    """Emit an exact-checked landing leg for `net_id`'s bare ball; returns
    (tip, [Segment, ...]) or None (no emission --
    ball not in a field, no gap, or the leg fails its exact check).

    Geometry (the fanout dog-bone jog, destination-side): 45-degree jog
    from the ball into the nearer inter-column gap (half a ball pitch to
    the side of the approach), straight along the exit axis past the
    field's outer row, tip half a pitch beyond the fringe.

    planned_pts (berth assignment): the pre-solved leg polyline is
    AUTHORITATIVE -- re-verified against the current board and emitted,
    but never replaced by a greedy pick (a fallback would resurrect the
    first-come contention the plan exists to remove)."""
    ctx = _ball_context(pcb_data, ball_xy)
    if ctx is None:
        return None
    pitch = ctx[1]
    bx, by = ball_xy
    if planned_pts is not None:
        pts = list(planned_pts)
        tip = pts[-1]
        if not _leg_clear(pcb_data, pts, config.layers[0],
                          config.track_width, config.clearance, net_id):
            # blocked by trunk copper the plan could not see: try this
            # member's own alternates (pre-filtered to conflict with no
            # sibling's planned berth), else dive.
            for alt in (alternate_pts or []):
                a = list(alt)
                if a == pts:
                    continue
                if _leg_clear(pcb_data, a, config.layers[0],
                              config.track_width, config.clearance,
                              net_id):
                    print(f"  Berth leg blocked; alternate berth: net "
                          f"{net_id} tip ({a[-1][0]:.2f},{a[-1][1]:.2f})")
                    pts, tip = a, a[-1]
                    break
            else:
                print(f"  Berth leg BLOCKED at emission: net {net_id} "
                      f"tip ({tip[0]:.2f},{tip[1]:.2f}) -- no leg (dive)")
                return None
    else:
        cands = _leg_candidates(ctx, ball_xy, approach_from,
                            grid=getattr(config, 'grid_step', 0.1))
        for _compat, _run, pts_c, tip in cands:
            pts = [p for i, p in enumerate(pts_c)
                   if i == 0 or math.hypot(p[0] - pts_c[i - 1][0],
                                           p[1] - pts_c[i - 1][1]) > 1e-6]
            if len(pts) < 2:
                continue
            if _leg_clear(pcb_data, pts, config.layers[0],
                          config.track_width, config.clearance, net_id):
                break
        else:
            return None
    layer = config.layers[0]
    width = config.track_width
    legs = getattr(pcb_data, '_emitted_legs', None)
    if legs is None:
        legs = []
        setattr(pcb_data, '_emitted_legs', legs)
    new_segs = []
    for (x1, y1), (x2, y2) in zip(pts, pts[1:]):
        seg = Segment(start_x=x1, start_y=y1, end_x=x2, end_y=y2,
                      width=width, layer=layer, net_id=net_id)
        pcb_data.segments.append(seg)
        legs.append(seg)
        new_segs.append(seg)
    print(f"  Exact landing leg: net {net_id} ball ({bx:.2f},{by:.2f}) "
          f"-> tip ({tip[0]:.2f},{tip[1]:.2f}) on {layer} "
          f"({len(pts) - 1} seg(s), pitch {pitch:.2f})")
    return tip, new_segs
