#!/usr/bin/env python3
"""Pre-flight canary: PROVE the injected congestion bites the feature path.

For each requested severity this tool
  1. verifies the cluster is clean (teardown + tc scrape on every node),
  2. measures baseline per-owner feature-fetch cost with netbench
     (real TCP pulls of N x d float32 rows from each owner over the private
     network, port 30050 — the firewall-trusted port),
  3. applies the impairment to the victim (broad eno1-root, via congestion2),
  4. re-measures, tears down, and
  5. reports kappa(owner) = t_slope_under / t_slope_baseline for every owner,
plus /sys/class/net/eno1 tx/rx byte-counter deltas per node per measurement
pass (evidence that the measured traffic actually traversed the shaped
interface).

PASS criteria (per severity): victim kappa >= --min-victim-kappa AND every
non-victim kappa within [1/tol, tol] (--flat-tol). Exit code 0 iff all pass.

This makes the congestion methodology *verified rather than asserted*: the
same per-row fetch cost the allocation/controller logic consumes is shown to
shift by the expected factor on the victim link only.

Run ON gnn1 (or with --ssh-mode public from outside; netbench client is then
executed on gnn1 over ssh).

Examples:
  python3 verify_congestion.py --cls c1 --rates 1000mbit,200mbit,50mbit \
      --victim gnn4 --out verification_c1.json
  python3 verify_congestion.py --cls c2 --streams 8 --victim gnn4 \
      --out verification_c2.json
  python3 verify_congestion.py --cls c4 --delay 15 --jitter 5 --loss 1 \
      --victim gnn4 --out verification_c4.json
"""

import argparse
import json
import os
import sys
import time

from congestion2 import (NODES, Cluster, Journal, apply_impairment,
                         build_iperf_client_cmd, build_iperf_server_cmd,
                         teardown_all)

NB_PORT = 30050  # firewall-trusted on the cluster (arbitrary high ports are not)


def _owners(victim):
    return [n for n in NODES if n != "gnn1"], victim


def start_netbench_servers(cluster, netbench, d):
    """Start a netbench server on every owner node (gnn2..4) with ssh -f
    semantics (detached; nohup alone does not survive the ssh channel)."""
    for n in [x for x in NODES if x != "gnn1"]:
        cluster.run(n, "pkill -9 -f 'netbench.py --server' 2>/dev/null || true",
                    action="nb_kill", check=False)
    for n in [x for x in NODES if x != "gnn1"]:
        cmd = (f"nohup $HOME/dt-venv/bin/python3 {netbench} --server --port {NB_PORT} --d {d} "
               f"> /tmp/netbench_server.log 2>&1 & sleep 0.5; "
               f"pgrep -f 'netbench.py --server' >/dev/null")
        cluster.run(n, cmd, action="nb_server", timeout=20)
    time.sleep(2)


def stop_netbench_servers(cluster):
    for n in [x for x in NODES if x != "gnn1"]:
        cluster.run(n, "pkill -9 -f 'netbench.py --server' 2>/dev/null || true",
                    action="nb_stop", check=False)


def read_nic_counters(cluster, iface):
    out = {}
    for n in NODES:
        _, txt = cluster.run(
            n, f"cat /sys/class/net/{iface}/statistics/tx_bytes "
               f"/sys/class/net/{iface}/statistics/rx_bytes",
            action="nic_counters", timeout=15)
        try:
            tx, rx = [int(x) for x in txt.split()]
        except ValueError:
            tx = rx = -1
        out[n] = {"tx_bytes": tx, "rx_bytes": rx}
    return out


def run_netbench_client(cluster, netbench, d, reps, out_path):
    """Run the netbench client ON gnn1 pulling from all owners; return the
    parsed per-host {t_init, t_slope} dict."""
    hosts = ",".join(NODES[n]["private"] for n in NODES if n != "gnn1")
    cmd = (f"$HOME/dt-venv/bin/python3 {netbench} --client --hosts {hosts} --port {NB_PORT} "
           f"--d {d} --reps {reps} --maxN 50000 --out {out_path}")
    cluster.run("gnn1", cmd, action="nb_client", timeout=600)
    _, txt = cluster.run("gnn1", f"cat {out_path}", action="nb_read",
                         timeout=15)
    return json.loads(txt)


def measure_pass(cluster, args, tag):
    """One measurement pass: NIC counters around a netbench client sweep."""
    pre = read_nic_counters(cluster, args.iface)
    nb = run_netbench_client(cluster, args.netbench, args.d, args.reps,
                             f"/tmp/nb_{tag}.json")
    post = read_nic_counters(cluster, args.iface)
    deltas = {n: {"tx_mb": round((post[n]["tx_bytes"] - pre[n]["tx_bytes"]) / 1e6, 1),
                  "rx_mb": round((post[n]["rx_bytes"] - pre[n]["rx_bytes"]) / 1e6, 1)}
              for n in NODES}
    # slope per private-ip -> node name
    ip2node = {NODES[n]["private"]: n for n in NODES}
    slopes = {ip2node.get(h, h): v["t_slope"] for h, v in nb.items()}
    return {"slopes_s_per_row": slopes, "nic_deltas": deltas, "raw": nb}


class _SevArgs:
    """Adapter so congestion2.apply_impairment sees the fields it expects."""
    def __init__(self, rate=None, delay=None, jitter=None, loss=None):
        self.rate, self.delay, self.jitter, self.loss = rate, delay, jitter, loss


def severity_list(args):
    if args.cls == "c1":
        return [("c1_" + r, _SevArgs(rate=r)) for r in args.rates.split(",")]
    if args.cls == "c4":
        return [("c4_delay%g_jit%g_loss%g" % (args.delay or 0, args.jitter or 0,
                                              args.loss or 0),
                 _SevArgs(delay=args.delay, jitter=args.jitter,
                          loss=args.loss))]
    if args.cls == "c2":
        return [("c2_%dstreams" % args.streams, None)]  # handled specially
    raise ValueError(args.cls)


def apply_c2_background(cluster, args, duration_s):
    """Continuous organic load covering the netbench measurement window."""
    victims = [args.victim]
    for v in victims:
        rc, _ = cluster.run(v, "which iperf3", action="check_iperf",
                            check=False)
        if rc != 0:
            raise RuntimeError(f"iperf3 missing on {v}")
        cluster.run(v, build_iperf_server_cmd(), action="iperf_server")
    procs = []
    senders = [n for n in NODES if n != args.victim]
    for s in senders if args.incast else senders[:1]:
        cmd = build_iperf_client_cmd(NODES[args.victim]["private"],
                                     streams=args.streams,
                                     duration_s=duration_s,
                                     direction="egress")
        procs.append(cluster.run_bg(s, cmd, action="iperf_bg"))
    return procs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cls", choices=["c1", "c2", "c4"], required=True)
    ap.add_argument("--victim", default="gnn4")
    ap.add_argument("--rates", default="1000mbit,500mbit,200mbit,100mbit,50mbit",
                    help="c1 severities")
    ap.add_argument("--delay", type=float, default=15.0)
    ap.add_argument("--jitter", type=float, default=5.0)
    ap.add_argument("--loss", type=float, default=1.0)
    ap.add_argument("--streams", type=int, default=8)
    ap.add_argument("--incast", action="store_true", default=True)
    ap.add_argument("--d", type=int, default=128, help="feature dim for netbench")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--netbench", default=os.path.expanduser("~/gdy2/netbench.py"))
    ap.add_argument("--iface", default="eno1")
    ap.add_argument("--ssh-mode", choices=["auto", "peer", "public"],
                    default="auto")
    ap.add_argument("--min-victim-kappa", type=float, default=2.0)
    ap.add_argument("--flat-tol", type=float, default=1.5)
    ap.add_argument("--out", default="verification.json")
    ap.add_argument("--journal", default="")
    args = ap.parse_args()

    journal = Journal(args.journal or None)
    cluster = Cluster(ssh_mode=args.ssh_mode, journal=journal,
                      iface=args.iface)
    report = {"cls": args.cls, "victim": args.victim, "t": time.time(),
              "severities": {}, "all_pass": True}
    try:
        teardown_all(cluster)
        start_netbench_servers(cluster, args.netbench, args.d)

        print("=== baseline pass (clean) ===")
        base = measure_pass(cluster, args, "base")
        report["baseline"] = base
        base_slopes = base["slopes_s_per_row"]

        for label, sev in severity_list(args):
            print(f"=== severity {label} ===")
            bg_procs = []
            if args.cls == "c2":
                # netbench sweep takes ~<120s at reps=5; keep load on longer
                bg_procs = apply_c2_background(cluster, args, duration_s=300)
                time.sleep(3)
            else:
                apply_impairment(cluster, args.cls, [args.victim], sev)

            under = measure_pass(cluster, args, label)

            if args.cls == "c2":
                for n in NODES:
                    cluster.run(n, "pkill -9 -x iperf3 2>/dev/null || true",
                                action="iperf_kill", check=False)
                for p in bg_procs:
                    try:
                        p.wait(timeout=5)
                    except Exception:
                        p.kill()
            else:
                teardown_all(cluster, [args.victim])

            kappas = {n: under["slopes_s_per_row"][n] / base_slopes[n]
                      for n in under["slopes_s_per_row"]}
            victim_k = kappas.get(args.victim, float("nan"))
            others = {n: k for n, k in kappas.items() if n != args.victim}
            ok_victim = victim_k >= args.min_victim_kappa
            ok_flat = all(1.0 / args.flat_tol <= k <= args.flat_tol
                          for k in others.values())
            passed = ok_victim and ok_flat
            report["severities"][label] = {
                "kappa": {n: round(k, 2) for n, k in kappas.items()},
                "victim_kappa": round(victim_k, 2),
                "victim_shift_ok": ok_victim,
                "others_flat_ok": ok_flat,
                "pass": passed,
                "nic_deltas": under["nic_deltas"],
            }
            report["all_pass"] = report["all_pass"] and passed
            others_str = {n: round(k, 2) for n, k in others.items()}
            print(f"  victim kappa={victim_k:.1f} "
                  f"(>= {args.min_victim_kappa}: {ok_victim}), "
                  f"others {others_str} flat: {ok_flat} "
                  f"-> {'PASS' if passed else 'FAIL'}")
    finally:
        stop_netbench_servers(cluster)
        teardown_all(cluster)
        journal.close()

    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"-> {args.out}  all_pass={report['all_pass']}")
    sys.exit(0 if report["all_pass"] else 1)


if __name__ == "__main__":
    main()
