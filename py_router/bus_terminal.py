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
from typing import List, Optional, Tuple

from kicad_parser import PCBData, Segment


def _pt_seg_d(px, py, x1, y1, x2, y2):
    dx, dy = x2 - x1, y2 - y1
    L2 = dx * dx + dy * dy
    t = 0.0 if L2 <= 0 else max(0.0, min(1.0, ((px - x1) * dx
                                               + (py - y1) * dy) / L2))
    return math.hypot(px - (x1 + t * dx), py - (y1 + t * dy))


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
            if (_pt_seg_d(s.start_x, s.start_y, x1, y1, x2, y2)
                    < s.width / 2.0 + hw + clearance - 1e-6
                    or _pt_seg_d(s.end_x, s.end_y, x1, y1, x2, y2)
                    < s.width / 2.0 + hw + clearance - 1e-6
                    or _pt_seg_d(x1, y1, s.start_x, s.start_y,
                                 s.end_x, s.end_y)
                    < s.width / 2.0 + hw + clearance - 1e-6
                    or _pt_seg_d(x2, y2, s.start_x, s.start_y,
                                 s.end_x, s.end_y)
                    < s.width / 2.0 + hw + clearance - 1e-6):
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


def emit_landing_leg(pcb_data: PCBData, net_id: int,
                     ball_xy: Tuple[float, float],
                     approach_from: Tuple[float, float],
                     config) -> Optional[Tuple[float, float]]:
    """Emit an exact-checked landing leg for `net_id`'s bare ball; returns
    (tip, [Segment, ...]) or None (no emission --
    ball not in a field, no gap, or the leg fails its exact check).

    Geometry (the fanout dog-bone jog, destination-side): 45-degree jog
    from the ball into the nearer inter-column gap (half a ball pitch to
    the side of the approach), straight along the exit axis past the
    field's outer row, tip half a pitch beyond the fringe."""
    bx, by = ball_xy
    # local ball pitch from the nearest neighbors, field extent from the
    # WHOLE owning footprint (a 3mm neighborhood capped the extent walk and
    # planted the tip INSIDE the field -- measured, f1_2).
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
    field = [p for p in owner.pads]
    ax, ay = approach_from[0] - bx, approach_from[1] - by

    def _leg_for(side, lsgn):
        """Candidate leg for one (exit side, gap lateral sign); returns
        (run_length, pts, tip) or None."""
        exit_x_axis = side in ('e', 'w')
        sgn = 1.0 if side in ('e', 's') else -1.0
        if exit_x_axis:
            gap_line = by + lsgn * pitch / 2.0
        else:
            gap_line = bx + lsgn * pitch / 2.0
        extent = bx if exit_x_axis else by
        for p in field:
            u = p.global_x if exit_x_axis else p.global_y
            w = p.global_y if exit_x_axis else p.global_x
            if abs(w - gap_line) < pitch * 0.75 and (u - extent) * sgn > 0:
                extent = u
        tip_u = extent + sgn * (pitch / 2.0 + 0.2)
        if exit_x_axis:
            jog = (bx + sgn * pitch / 2.0, gap_line)
            tip = (tip_u, gap_line)
        else:
            jog = (gap_line, by + sgn * pitch / 2.0)
            tip = (gap_line, tip_u)
        run = abs(tip_u - (bx if exit_x_axis else by))
        return run, [(bx, by), jog, tip], tip

    # Side order: APPROACH-COMPATIBLE first (the tip must face the arriving
    # pack -- nearest-boundary alone sent legs out the far side and the
    # route had to wrap the band, 1 failed, measured f2_2), then shortest
    # run. Per side, the gap lateral toward the approach first.
    _svec = {'e': (1.0, 0.0), 'w': (-1.0, 0.0),
             's': (0.0, 1.0), 'n': (0.0, -1.0)}
    cands = []
    for side in ('e', 'w', 's', 'n'):
        if side in ('e', 'w'):
            pref = 1.0 if ay >= 0 else -1.0
        else:
            pref = 1.0 if ax >= 0 else -1.0
        compat = (_svec[side][0] * ax + _svec[side][1] * ay) <= 0
        for lsgn in (pref, -pref):
            c = _leg_for(side, lsgn)
            if c is not None:
                cands.append((compat,) + c)
    cands.sort(key=lambda t: (t[0], t[1]))
    cands = [c[1:] for c in cands]
    for _run, pts_c, tip in cands:
        pts = [p for i, p in enumerate(pts_c)
               if i == 0 or math.hypot(p[0] - pts_c[i - 1][0],
                                       p[1] - pts_c[i - 1][1]) > 1e-6]
        if len(pts) < 2:
            continue
        if _leg_clear(pcb_data, pts, config.layers[0], config.track_width,
                      config.clearance, net_id):
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
