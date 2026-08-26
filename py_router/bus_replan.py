#!/usr/bin/env python3
"""#622 part B: learn a bus plan FROM A ROUTED BOARD (chain-level replan).

The routed board is the one artifact that holds the full negotiated
outcome -- which layer each member actually ran on, the lateral order
the pack actually settled into, and the corridor the copper actually
took. This tool reads those facts and emits a plan JSON; a fresh route
of the ORIGINAL board consumes it via KICAD_BUS_PLAN_FILE, so round
N+1 plans with round N's knowledge instead of probes. Chain-level
only: no in-run non-regression machinery -- the old routed file is
still there, callers compare and keep the better board. Usable N
times (route -> analyze -> route -> ...).

Pinned by the plan (user decision 2c): river layers, member physical
order, corridor reference paths. Berths RE-SOLVE fresh each round.

Usage:
  python3 py_router/bus_replan.py routed.kicad_pcb -o plan.json \
      --nets "/DDR3 16x1/SA12" ...       # the bus scope (as routed)
"""
import argparse
import json
import math
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kicad_parser import parse_kicad_pcb


def trace_net_path(pcb_data, net_id, src, tgt):
    """Dijkstra over the AUTHORITATIVE copper graph -- the same one
    check_connected builds (check_net_connectivity return_graph=True),
    with all its credit rules: T-joints, via barrels, through-pad
    islands, zones. Point ids: segment i's endpoints are 2i/2i+1;
    pads/vias have representative ids in pad_index_repr/via_index_repr.
    Traces from the pad nearest `src` to the pad nearest `tgt`.
    Returns [(x_mm, y_mm, layer_name)] or None."""
    import heapq
    from check_connected import check_net_connectivity
    segs = [s for s in pcb_data.segments if s.net_id == net_id]
    vias = [v for v in pcb_data.vias if v.net_id == net_id]
    pads = pcb_data.pads_by_net.get(net_id, [])
    zones = [z for z in (getattr(pcb_data, 'zones', None) or [])
             if getattr(z, 'net_id', None) == net_id]
    if not segs or len(pads) < 2:
        return None
    res = check_net_connectivity(net_id, segs, vias, pads, zones=zones,
                                 return_graph=True, pcb_data=pcb_data)
    g = res.get('graph') or {}
    edges = g.get('edges') or []
    if not edges:
        return None
    coord = {}
    lay = {}
    for i, s in enumerate(segs):
        coord[2 * i] = (s.start_x, s.start_y)
        coord[2 * i + 1] = (s.end_x, s.end_y)
        lay[2 * i] = lay[2 * i + 1] = s.layer
    for vi, pid in (g.get('via_index_repr') or {}).items():
        coord.setdefault(pid, (vias[vi].x, vias[vi].y))
    pad_repr = g.get('pad_index_repr') or {}
    for pi, pid in pad_repr.items():
        coord.setdefault(pid, (pads[pi].global_x, pads[pi].global_y))
    adj = defaultdict(list)
    for a, b in edges:
        ca, cb = coord.get(a), coord.get(b)
        w = (math.hypot(ca[0] - cb[0], ca[1] - cb[1])
             if ca and cb else 0.01)
        adj[a].append((b, w))
        adj[b].append((a, w))
    if not pad_repr:
        return None

    def pad_node(pt):
        pi = min(pad_repr,
                 key=lambda i: math.hypot(pads[i].global_x - pt[0],
                                          pads[i].global_y - pt[1]))
        return pad_repr[pi]

    start, goal = pad_node(src), pad_node(tgt)
    dist = {start: 0.0}
    prev = {}
    pq = [(0.0, start)]
    while pq:
        d, u = heapq.heappop(pq)
        if u == goal:
            break
        if d > dist.get(u, float('inf')):
            continue
        for w, c in adj[u]:
            nd = d + c
            if nd < dist.get(w, float('inf')):
                dist[w] = nd
                prev[w] = u
                heapq.heappush(pq, (nd, w))
    if goal not in dist:
        return None
    ids = [goal]
    while ids[-1] != start:
        ids.append(prev[ids[-1]])
    ids.reverse()
    # layer per point: segment endpoints know theirs; pad/via points
    # take the NEXT known layer, then the last known -- so the path
    # geometry includes its terminal pads (the pad's union edge can
    # enter a one-segment landing leg at its FAR end, which otherwise
    # silently dropped the ball-to-tip run)
    known = [lay.get(pid) for pid in ids]
    nxt = None
    for i in range(len(ids) - 1, -1, -1):
        if known[i] is not None:
            nxt = known[i]
        elif nxt is not None:
            known[i] = nxt
    out = []
    last_layer = None
    for pid, L in zip(ids, known):
        c = coord.get(pid)
        if c is None:
            continue                     # zone blob point, no geometry
        if L is None:
            L = last_layer
        if L is None:
            continue
        if out and math.hypot(c[0] - out[-1][0],
                              c[1] - out[-1][1]) < 1e-6:
            continue
        out.append((c[0], c[1], L))
        last_layer = L
    return out if len(out) >= 2 else None


def path_layer_lengths(path):
    per = defaultdict(float)
    for (x1, y1, l1), (x2, y2, l2) in zip(path, path[1:]):
        if l1 == l2:
            per[l1] += math.hypot(x2 - x1, y2 - y1)
    return per


def resample(path_xy, step=0.2):
    """Uniform-arc resample of an (x, y) polyline."""
    if len(path_xy) < 2:
        return list(path_xy)
    arc = [0.0]
    for a, b in zip(path_xy, path_xy[1:]):
        arc.append(arc[-1] + math.hypot(b[0] - a[0], b[1] - a[1]))
    total = arc[-1]
    if total < 1e-9:
        return [path_xy[0]]
    out = []
    s = 0.0
    i = 0
    while s < total:
        while arc[i + 1] < s:
            i += 1
        span = arc[i + 1] - arc[i]
        f = 0.0 if span < 1e-12 else (s - arc[i]) / span
        out.append((path_xy[i][0]
                    + f * (path_xy[i + 1][0] - path_xy[i][0]),
                    path_xy[i][1]
                    + f * (path_xy[i + 1][1] - path_xy[i][1])))
        s += step
    out.append(path_xy[-1])
    return out


def lateral_offset(path_xy, center_xy):
    """Mean signed lateral offset of `path_xy` from the centerline
    (sign via the centerline's local normal)."""
    if len(center_xy) < 2 or not path_xy:
        return 0.0
    tot = n = 0
    for cx, cy in center_xy[:: max(1, len(center_xy) // 24)]:
        best = None
        for px, py in path_xy:
            d2 = (px - cx) ** 2 + (py - cy) ** 2
            if best is None or d2 < best[0]:
                best = (d2, px, py)
        # local tangent at this center station
        k = min(range(len(center_xy)),
                key=lambda i: (center_xy[i][0] - cx) ** 2
                + (center_xy[i][1] - cy) ** 2)
        k2 = min(k + 1, len(center_xy) - 1)
        k1 = max(k - 1, 0)
        tx, ty = (center_xy[k2][0] - center_xy[k1][0],
                  center_xy[k2][1] - center_xy[k1][1])
        L = math.hypot(tx, ty)
        if L < 1e-9:
            continue
        nx, ny = -ty / L, tx / L
        tot += (best[1] - cx) * nx + (best[2] - cy) * ny
        n += 1
    return tot / n if n else 0.0


def analyze_board(pcb_data, net_names):
    """Read the realized routing facts per net: dominant layer, the
    traced copper path, and a lateral ORDER KEY within the net's
    (component-pair, layer) river. Emits a PER-NET plan (schema v2):
    the round-2 consumer -- which knows its own detected groups --
    buckets members by these facts, so the plan survives any grouping
    disagreement between the routed board and a fresh detection."""
    name_to_id = {net.name: nid for nid, net in pcb_data.nets.items()}
    ids = [name_to_id[n] for n in net_names if n in name_to_id]
    missing = [n for n in net_names if n not in name_to_id]
    if missing:
        print(f"warning: {len(missing)} net(s) not on board: "
              f"{missing[:4]}")
    traced = {}
    ends = {}
    comp_pair = {}
    for nid in ids:
        pads = pcb_data.pads_by_net.get(nid, [])
        if len(pads) < 2:
            continue
        best = None
        for i in range(len(pads)):
            for j in range(i + 1, len(pads)):
                d = math.hypot(pads[i].global_x - pads[j].global_x,
                               pads[i].global_y - pads[j].global_y)
                if best is None or d > best[0]:
                    best = (d, pads[i], pads[j])
        _, pa, pb = best
        p = trace_net_path(pcb_data, nid,
                           (pa.global_x, pa.global_y),
                           (pb.global_x, pb.global_y))
        if p and len(p) >= 2:
            traced[nid] = p
            ends[nid] = ((pa.global_x, pa.global_y),
                         (pb.global_x, pb.global_y))
            comp_pair[nid] = tuple(sorted((pa.component_ref,
                                           pb.component_ref)))
    # dominant layer per net, judged on the MID-RUN only: the terminal
    # 3mm at each end is forced escape geometry (dogbone teeth, landing
    # legs, dives under the ball field), not the river -- counting it
    # misclassifies a member whose middle rides one layer but whose
    # ends dove (measured r2_51: whole-path dominance moved 3 members
    # across rivers vs their round-1 assignment and cost completion).
    dom = {}
    layer_mm = {}
    for nid, p in traced.items():
        s_pt, t_pt = ends[nid]
        mid = [q for q in p
               if math.hypot(q[0] - s_pt[0], q[1] - s_pt[1]) > 3.0
               and math.hypot(q[0] - t_pt[0], q[1] - t_pt[1]) > 3.0]
        per = path_layer_lengths(mid if len(mid) >= 2 else p)
        if not per:
            per = path_layer_lengths(p)  # short net: no mid-run at all
        if per:
            dom[nid] = max(per, key=per.get)
            layer_mm[nid] = {k: round(v, 2) for k, v in per.items()}
    rivers = defaultdict(list)
    for nid in traced:
        if nid in dom:
            rivers[(comp_pair[nid], dom[nid])].append(nid)
    order_key = {}
    for (_, lay), members in rivers.items():
        med = sorted(members,
                     key=lambda nid: sum(
                         math.hypot(b[0] - a[0], b[1] - a[1])
                         for a, b in zip(traced[nid], traced[nid][1:]))
                     )[len(members) // 2]
        center_xy = resample([(p[0], p[1]) for p in traced[med]])
        for nid in members:
            order_key[nid] = lateral_offset(
                [(p[0], p[1]) for p in traced[nid]], center_xy)
    plan = {'version': 2,
            'source_board': getattr(pcb_data, 'source_path', ''),
            'nets': {}}
    for nid in traced:
        if nid not in dom:
            continue
        plan['nets'][pcb_data.nets[nid].name] = {
            'layer': dom[nid],
            'layers_mm': layer_mm.get(nid, {}),
            'order_key': round(order_key.get(nid, 0.0), 4),
            'path': [[round(p[0], 3), round(p[1], 3), p[2]]
                     for p in traced[nid]],
        }
    return plan


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('board')
    ap.add_argument('-o', '--out', required=True)
    ap.add_argument('--nets', nargs='+', required=True)
    a = ap.parse_args()
    pcb = parse_kicad_pcb(a.board)
    plan = analyze_board(pcb, a.nets)
    with open(a.out, 'w', encoding='utf-8') as f:
        json.dump(plan, f, indent=1)
    by_layer = defaultdict(list)
    for nm, rec in plan['nets'].items():
        by_layer[rec['layer']].append((rec['order_key'], nm))
    print(f"bus plan: {len(plan['nets'])} net(s) -> {a.out}")
    for lay, rows in sorted(by_layer.items()):
        rows.sort()
        print(f"  {lay} ({len(rows)}): "
              f"{' '.join(nm.split('/')[-1] for _k, nm in rows)}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
