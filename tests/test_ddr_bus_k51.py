#!/usr/bin/env python3
"""#622 K=51 chain gate: route the committed DDR bus bench with the
full take-2 stack and hold the campaign's acceptance line.

The bench (kicad_files/ddr_bus_bench.kicad_pcb, built by
tests/build_ddr_bus_bench.py) plus this env-armed stack produced the
campaign's first fully-connected K=51 (231 vias / 0 failed / DRC clean
up to one same-net via-drill nit). This test pins that outcome:

  - failed == 0 (completion is the doctrine's first axis),
  - vias within a wobble band (<= 280; measured 231, single-board
    wobble is real so the band is generous -- a regression that costs
    completion or +50 vias is a mechanism break, not wobble),
  - ZERO commit-time crossing-canary hits (KICAD_DBG_CROSS_CANARY:
    every same-layer different-net crossing laid by any commit path),
  - no DIFFERENT-net DRC violations (same-net drill nits are the known
    stacked-copper class, tracked separately).

Classified integration by the subprocess import (run_all --fast skips
it; full run_all and direct invocation run it, ~3-4 minutes).
"""
import json
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BENCH = os.path.join(ROOT, 'kicad_files', 'ddr_bus_bench.kicad_pcb')

NETS = ('SA12 SDQM1 SDQ9 SDQ10 SDQ11 SDQ8 SDQ12 SDQ13 SDQM0 SDQ14 SA14 '
        'SDQ15 SDQ0 SA11 SDQ2 SDQ1 SA15 SA10 SDQ4 SA0 SDQ5 SBA1 SDQ7 '
        'SDQ3 SA1 SA3 SDQ6 SA4 SA2 SCS0 SCS1 SA8 SCKE1 SA5 SA6 SA9 SBA2 '
        'SCAS SCKE0 SODT1 SRAS SRST SA13 SA7 SBA0 SODT0 SWE SVREF SZQ '
        'DU1_ZQ DU1_ZQ1').split()

ENV = {
    'KICAD_BUS_EXACT_LEGS': '1', 'KICAD_BUS_BERTHS': '1',
    'KICAD_BUS_FOLLOW_LEADER': '1', 'KICAD_BUS_OFFLANE_MULT': '3',
    'KICAD_ATTRACT_POTENTIAL': '10000', 'KICAD_ATTRACT_BASIN_PCT': '50',
    'KICAD_BUS_MACRO_RIVERS': '1', 'KICAD_BUS_EXIT_EXT': '0.8',
    'KICAD_DBG_CROSS_CANARY': '1',
}
VIA_BAND = 280


def main():
    if not os.path.exists(BENCH):
        print(f"BROKEN: bench missing ({BENCH}); run "
              f"tests/build_ddr_bus_bench.py first")
        return 2
    tmp = tempfile.mkdtemp(prefix='ddr_bus_k51_')
    out = os.path.join(tmp, 'k51.kicad_pcb')
    env = dict(os.environ)
    env.update(ENV)
    argv = [sys.executable, '-X', 'utf8',
            os.path.join(ROOT, 'py_router', 'route.py'), BENCH, out,
            '--bus', '--nets'] + [f'/DDR3 16x1/{n}' for n in NETS] + [
            '--ordering', 'bus', '--bus-attraction-radius', '2.5',
            '--clearance', '0.1', '--track-width', '0.1',
            '--via-size', '0.25', '--via-drill', '0.15',
            '--hole-to-hole-clearance', '0.2', '--max-ripup', '5',
            '--no-bga-zone']
    r = subprocess.run(argv, capture_output=True, text=True, env=env,
                       timeout=900)
    mins = re.findall(r'JSON_SUMMARY_MIN: (\{.*\})', r.stdout)
    if not mins:
        print("BROKEN: route emitted no JSON_SUMMARY_MIN "
              f"(exit {r.returncode})")
        print(r.stdout[-2000:])
        print(r.stderr[-1000:])
        return 2
    verdict = json.loads(mins[-1])
    if verdict.get('scope') != 'merged':
        print(f"BROKEN: last JSON_SUMMARY_MIN scope is "
              f"{verdict.get('scope')!r}, not 'merged'")
        return 2
    failed = verdict.get('failed')
    vias = verdict.get('vias')
    canary = r.stdout.count('[cross-canary]')
    print(f"chain: failed={failed} vias={vias} canary={canary}")
    ok = True
    if failed != 0:
        print(f"FAIL: {failed} net(s) not fully routed "
              f"({verdict.get('failed_single')})")
        ok = False
    if vias is None or vias > VIA_BAND:
        print(f"FAIL: vias {vias} above the {VIA_BAND} band")
        ok = False
    if canary:
        print(f"FAIL: {canary} commit-time crossing(s) laid")
        ok = False

    d = subprocess.run([sys.executable,
                        os.path.join(ROOT, 'py_router', 'check_drc.py'),
                        out, '--clearance-margin', '0.1'],
                       capture_output=True, text=True, timeout=300)
    cross_net = []
    for ln in d.stdout.split('\n'):
        m = re.match(r'\s*(?:Via:)?(/[^<]+?)\s*<->\s*'
                     r'(?:Via:)?(/.+?)(?:\s*\(.*\))?$', ln)
        if m and m.group(1).strip() != m.group(2).strip():
            cross_net.append(ln.strip())
    if cross_net:
        print(f"FAIL: {len(cross_net)} different-net DRC violation(s):")
        for ln in cross_net[:5]:
            print(f"  {ln}")
        ok = False
    elif d.returncode != 0:
        print("note: DRC exit nonzero but only same-net items "
              "(stacked-copper class, tracked separately)")
    print('DDR-BUS-K51', 'PASS' if ok else 'FAIL')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
