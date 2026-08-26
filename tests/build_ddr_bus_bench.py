#!/usr/bin/env python3
"""Build (and verify) the #622 DDR bus microscope bench.

The bench is the take-2 campaign's acceptance vehicle: a two-BGA DDR3
board (U1 controller, DU1 DRAM) where

  - U1 gets a PLANNED dogbone fanout: the pre-fanout probe plan
    (KICAD_FANOUT_PROBE_PLAN=alloc) arms bus identity before stubs
    exist, per-group layer assignment, grant-or-natural side
    re-allocation, lane-run surface escapes and the dive rescue --
    measured: this bench's planned fanout turned the same routing
    stack's K=51 from 241 vias / 1 stranded into 231 / 0 (the
    campaign's first fully-connected full-bus run);
  - DU1 stays BARE (user law: no fanout on the small BGA; the route
    step's exact-emitted landing legs are the destination fanout);
  - the GND pour is STRIPPED: pours are soft during routing and the
    fragility field was measured to be the pour's only in-route effect
    (zone-delete reproduced KICAD_PLANE_FRAGILITY_COST=0 bit-for-bit),
    so the pour-free board is the honest like-for-like bench against
    the pour-less human reference.

Input:  kicad_files/ddr_bus_bench_src.kicad_pcb  (post-planes chain
        stage of the fake board; the generator lives outside the repo,
        so the source board is committed)
Output: kicad_files/ddr_bus_bench.kicad_pcb (+ .kicad_pro sibling)

Run it directly (not under run_all -- the fanout step costs ~a minute):

    python3 tests/build_ddr_bus_bench.py [--out DIR]

It rebuilds from the committed source and VERIFIES the result (escape
count, zero zones, copper-count band, octilinearity of the fanout's
lane runs is NOT asserted -- the dogbone connectors are off-grid by
design), then writes the bench next to the source. Routing is
deterministic but python-version effects exist (math.fsum lore), so
counts are asserted as bands, not exact.
"""
import argparse
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(ROOT, 'kicad_files', 'ddr_bus_bench_src.kicad_pcb')
sys.path.insert(0, os.path.join(ROOT, 'py_router'))

FANOUT_ARGS = [
    '--component', 'U1', '--layers', 'F.Cu', 'B.Cu',
    '--nets', '*', '!GND', '!VCC-DRAM',
    '--diff-pairs', '*SCKP*', '*SCKN*', '*SDQS0P*', '*SDQS0N*',
    '*SDQS1P*', '*SDQS1N*',
    '--diff-pair-gap', '0.1143', '--escape-method', 'dogbone',
    '--track-width', '0.1', '--clearance', '0.1',
    '--via-size', '0.2', '--via-drill', '0.1',
]


def strip_zones(text: str):
    out, i, removed = [], 0, 0
    while True:
        j = text.find('(zone', i)
        if j < 0:
            out.append(text[i:])
            break
        out.append(text[i:j])
        depth, k = 0, j
        while k < len(text):
            if text[k] == '(':
                depth += 1
            elif text[k] == ')':
                depth -= 1
                if depth == 0:
                    break
            k += 1
        removed += 1
        i = k + 1
    return ''.join(out), removed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(ROOT, 'kicad_files'),
                    help='output directory (default: kicad_files)')
    args = ap.parse_args()
    if not os.path.exists(SRC):
        print(f"BROKEN: source board missing: {SRC}")
        return 2
    tmp_fan = os.path.join(args.out, 'ddr_bus_bench_fanout_tmp.kicad_pcb')
    tmp_pl = os.path.join(args.out, 'ddr_bus_bench_placed_tmp.kicad_pcb')
    out_pcb = os.path.join(args.out, 'ddr_bus_bench.kicad_pcb')
    out_pro = os.path.join(args.out, 'ddr_bus_bench.kicad_pro')

    env = dict(os.environ)
    env['KICAD_FANOUT_PROBE_PLAN'] = 'alloc'
    r = subprocess.run(
        [sys.executable, '-X', 'utf8',
         os.path.join(ROOT, 'py_router', 'bga_fanout.py'), SRC,
         '--output', tmp_fan] + FANOUT_ARGS,
        capture_output=True, text=True, env=env)
    m = re.search(r'"escaped": (\d+), "failed": (\d+)', r.stdout)
    if not m:
        print("BROKEN: fanout emitted no escape tally")
        print(r.stdout[-2000:])
        print(r.stderr[-2000:])
        return 2
    escaped, fan_failed = int(m.group(1)), int(m.group(2))
    # 55/56 measured (SVREF's shared-pad geometry declines); a real
    # regression shows up as a multi-net drop.
    if escaped < 54 or fan_failed > 2:
        print(f"FAIL: fanout escaped {escaped}, failed {fan_failed} "
              f"(expected >=54 / <=2)")
        return 1

    r2 = subprocess.run(
        [sys.executable, '-X', 'utf8',
         os.path.join(ROOT, 'py_router', 'place_fanout_clearance.py'),
         tmp_fan, tmp_pl, '--clearance', '0.1'],
        capture_output=True, text=True)
    if not os.path.exists(tmp_pl):
        print("BROKEN: place_fanout_clearance produced no board")
        print(r2.stdout[-1500:])
        return 2

    text = open(tmp_pl, encoding='utf-8').read()
    text, removed = strip_zones(text)
    with open(out_pcb, 'w', encoding='utf-8') as f:
        f.write(text)
    # never a board without its .kicad_pro (#441)
    pro_src = tmp_pl.replace('.kicad_pcb', '.kicad_pro')
    if not os.path.exists(pro_src):
        pro_src = SRC.replace('.kicad_pcb', '.kicad_pro')
    with open(pro_src, encoding='utf-8') as f:
        pro = f.read()
    with open(out_pro, 'w', encoding='utf-8') as f:
        f.write(pro)

    from kicad_parser import parse_kicad_pcb
    pcb = parse_kicad_pcb(out_pcb)
    n_seg, n_via, n_zone = len(pcb.segments), len(pcb.vias), len(pcb.zones)
    ok = (n_zone == 0 and 180 <= n_seg <= 320 and 20 <= n_via <= 60)
    print(f"bench: escapes {escaped}/{escaped + fan_failed}, zones "
          f"removed {removed}, segments {n_seg}, vias {n_via}, "
          f"zones {n_zone}")
    for t in (tmp_fan, tmp_fan.replace('.kicad_pcb', '.kicad_pro'),
              tmp_pl, pro_src if 'tmp' in pro_src else None):
        if t and os.path.exists(t) and 'tmp' in os.path.basename(t):
            os.remove(t)
    if not ok:
        print("FAIL: bench outside expected bands "
              "(zones==0, segs 180-320, vias 20-60)")
        return 1
    print(f"BENCH OK: {out_pcb}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
