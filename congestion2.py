#!/usr/bin/env python3
"""Congestion injection v2 — broad, verified, journaled.

Replaces the legacy port-30050-scoped netem driver (congestion.py). Rationale,
validated on the cluster 2026-06-29: DGL's feature DATA flows over EPHEMERAL
high ports, so any port-scoped tc rule misses the data plane entirely (it only
perturbs the RPC control port). All injection here is therefore applied to the
ROOT of the private interface (eno1). eno1 carries ONLY the private cluster
subnet (10.52.0.0/22); management SSH arrives on the public interface, so root
shaping of eno1 is subnet-scoped by construction and cannot lock us out.

Congestion classes
  c1  bandwidth contention  : tbf rate-limit on victim egress (proven mechanism,
                              measured kappa 9.6 -> 191 for 1000 -> 50 mbit)
  c2  organic cross-traffic : iperf3 flows from the other nodes saturating the
                              victim's link (duty-cycled; real TCP dynamics —
                              queueing, cwnd, statistical multiplexing). With
                              --direction egress (default) clients use -R so the
                              VICTIM transmits, contending with feature serving.
                              --incast fires all senders simultaneously.
  c4  delay/jitter/loss     : netem delay X ms jitter Y ms (normal distribution)
                              + Gilbert-Elliott loss (gemodel)

Modes
  steady     : apply once; exposure identical across methods by construction.
  squarewave : wall-clock period/duty (NOT epoch-log tailing — methods differ in
               epoch time, so wall-clock is the only method-agnostic schedule).
               Period, phase and every transition are journaled, so realized
               exposure is verifiable from timestamps, not asserted.

Every ssh/tc invocation's return code is checked; failures abort loudly.
All transitions are appended to a JSONL journal: {t, node, action, cmd, rc}.
Teardown is idempotent and verified (tc qdisc show scraped on every node).

Usage (run on gnn1 by default; --ssh-mode public to drive from outside):
  python3 congestion2.py apply    --cls c1 --victims gnn4 --rate 200mbit
  python3 congestion2.py run      --cls c1 --mode squarewave --rate 200mbit \
                                  --victims gnn4 --duration 1800 --period 120 \
                                  --duty 0.5 --journal cong_journal.jsonl
  python3 congestion2.py run      --cls c2 --victims gnn4 --duration 1800 \
                                  --on 30 --off 30 --streams 8 --journal j.jsonl
  python3 congestion2.py teardown
  python3 congestion2.py verify
  python3 congestion2.py status
"""

import argparse
import json
import os
import shlex
import signal
import socket
import subprocess
import sys
import time

# ---------------------------------------------------------------------------
# Cluster registry
# ---------------------------------------------------------------------------

NODES = {
    "gnn1": {"public": "129.114.108.218", "private": "10.52.2.119"},
    "gnn2": {"public": "129.114.108.186", "private": "10.52.3.217"},
    "gnn3": {"public": "129.114.108.115", "private": "10.52.3.123"},
    "gnn4": {"public": "129.114.108.101", "private": "10.52.3.89"},
}
PEER_KEY = "~/.ssh/peerkey.pem"      # exists on gnn1 for node-to-node
PUBLIC_KEY = "~/.ssh/newtacckey.pem"  # local WSL / laptop
IPERF_PORT = 5201
DEFAULT_IFACE = "eno1"

DIRTY_QDISCS = ("tbf", "netem", "htb", "prio")


# ---------------------------------------------------------------------------
# Pure functions (unit-tested, no network)
# ---------------------------------------------------------------------------

def build_tc_cmd(cls_, iface=DEFAULT_IFACE, rate=None, delay_ms=None,
                 jitter_ms=None, loss_pct=None):
    """Build the shell command that installs the impairment on one node.

    del-then-add (legacy-proven pattern): the default root qdisc is mq, which
    must be removed before a classless root qdisc can be installed.
    """
    del_part = f"sudo tc qdisc del dev {iface} root 2>/dev/null || true"
    if cls_ == "c1":
        if not rate:
            raise ValueError("c1 requires --rate (e.g. 200mbit)")
        add = (f"sudo tc qdisc add dev {iface} root tbf rate {rate} "
               f"burst 32kbit latency 400ms")
    elif cls_ == "c4":
        opts = []
        if delay_ms:
            j = f" {jitter_ms}ms distribution normal" if jitter_ms else ""
            opts.append(f"delay {delay_ms}ms{j}")
        if loss_pct:
            # Gilbert-Elliott: p (good->bad) = loss_pct, r (bad->good) = 25%
            opts.append(f"loss gemodel {loss_pct}% 25%")
        if not opts:
            raise ValueError("c4 requires --delay and/or --loss")
        add = f"sudo tc qdisc add dev {iface} root netem {' '.join(opts)}"
    else:
        raise ValueError(f"build_tc_cmd: unknown/unsupported class {cls_!r}")
    return f"{del_part}; {add}"


def teardown_cmds(iface=DEFAULT_IFACE):
    """Idempotent per-node teardown command list (safe to run repeatedly)."""
    return [
        f"sudo tc qdisc del dev {iface} root 2>/dev/null || true",
        "pkill -9 -x iperf3 2>/dev/null || true",
    ]


def tc_state_is_clean(tc_show_output):
    """True iff `tc qdisc show dev IFACE` output contains no impairment."""
    low = tc_show_output.lower()
    return not any(q in low for q in DIRTY_QDISCS)


def build_iperf_client_cmd(victim_private_ip, port=IPERF_PORT, streams=8,
                           duration_s=30, direction="egress"):
    """iperf3 client burst. direction=egress => -R: the victim (server side)
    transmits, saturating the same egress path that serves features."""
    rev = " -R" if direction == "egress" else ""
    return (f"iperf3 -c {victim_private_ip} -p {port} -P {streams} "
            f"-t {duration_s}{rev} --connect-timeout 5000 >/dev/null 2>&1")


def build_iperf_server_cmd(port=IPERF_PORT):
    # Idempotence via a listening-port check (ss), NOT pgrep: a pgrep guard
    # matches ANY iperf3 process, so it silently suppresses the additional
    # per-port servers that a real incast needs (and `pgrep -f` would match
    # this command's own ssh argv).
    return (f"ss -ltn | grep -q ':{port} ' || iperf3 -s -p {port} -D")


def squarewave_schedule(duration_s, period_s, duty, phase_s=0.0):
    """[(t_on, t_off), ...] offsets from run start. ON during the first
    duty*period of every period, starting at phase_s."""
    if period_s <= 0 or not (0.0 < duty <= 1.0):
        raise ValueError("period must be >0 and 0<duty<=1")
    windows = []
    k = 0
    on_len = period_s * duty
    while True:
        s = k * period_s + phase_s
        if s >= duration_s:
            break
        e = min(s + on_len, duration_s)
        if e > s >= 0:
            windows.append((s, e))
        k += 1
    return windows


def exposure_fraction(schedule, t0, t1):
    """Fraction of [t0, t1] covered by ON windows of `schedule`."""
    if t1 <= t0:
        return 0.0
    covered = sum(max(0.0, min(e, t1) - max(s, t0)) for s, e in schedule)
    return covered / (t1 - t0)


# ---------------------------------------------------------------------------
# Journal
# ---------------------------------------------------------------------------

class Journal:
    def __init__(self, path=None):
        self.path = path
        self._fh = open(path, "a") if path else None

    def log(self, **kw):
        kw.setdefault("t", time.time())
        line = json.dumps(kw, sort_keys=True)
        print(f"[journal] {line}", flush=True)
        if self._fh:
            self._fh.write(line + "\n")
            self._fh.flush()

    def close(self):
        if self._fh:
            self._fh.close()
            self._fh = None


# ---------------------------------------------------------------------------
# SSH execution layer
# ---------------------------------------------------------------------------

class Cluster:
    def __init__(self, ssh_mode="auto", journal=None, iface=DEFAULT_IFACE):
        self.iface = iface
        self.journal = journal or Journal(None)
        host = socket.gethostname().split(".")[0]
        if ssh_mode == "auto":
            ssh_mode = "peer" if host in NODES else "public"
        self.ssh_mode = ssh_mode
        self.local_host = host if host in NODES else None

    def _ssh_prefix(self, node):
        if self.ssh_mode == "peer":
            key, addr = PEER_KEY, NODES[node]["private"]
        else:
            key, addr = PUBLIC_KEY, NODES[node]["public"]
        return (f"ssh -i {key} -o StrictHostKeyChecking=no "
                f"-o ConnectTimeout=10 cc@{addr}")

    def run(self, node, cmd, action="cmd", timeout=30, check=True):
        """Run `cmd` on `node` (locally if node == this host). Journal + rc check."""
        if node == self.local_host:
            full = cmd
        else:
            full = f"{self._ssh_prefix(node)} {shlex.quote(cmd)}"
        try:
            r = subprocess.run(full, shell=True, capture_output=True,
                               text=True, timeout=timeout)
            rc, out = r.returncode, (r.stdout + r.stderr).strip()
        except subprocess.TimeoutExpired:
            rc, out = 124, "TIMEOUT"
        self.journal.log(node=node, action=action, cmd=cmd, rc=rc)
        if check and rc != 0:
            self.journal.log(node=node, action="ABORT",
                             cmd=cmd, rc=rc, out=out[:500])
            raise RuntimeError(
                f"congestion2: command failed on {node} (rc={rc}): {cmd}\n{out[:500]}")
        return rc, out

    def run_bg(self, node, cmd, action="bg"):
        """Fire-and-track a background burst (used for parallel iperf3 clients)."""
        if node == self.local_host:
            full = cmd
        else:
            full = f"{self._ssh_prefix(node)} {shlex.quote(cmd)}"
        p = subprocess.Popen(full, shell=True,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.journal.log(node=node, action=action, cmd=cmd, rc=None)
        return p


# ---------------------------------------------------------------------------
# High-level operations
# ---------------------------------------------------------------------------

def apply_impairment(cluster, cls_, victims, args):
    for v in victims:
        cmd = build_tc_cmd(cls_, iface=cluster.iface, rate=args.rate,
                           delay_ms=args.delay, jitter_ms=args.jitter,
                           loss_pct=args.loss)
        cluster.run(v, cmd, action=f"apply_{cls_}")
        # verify installed
        _, out = cluster.run(v, f"tc qdisc show dev {cluster.iface}",
                             action="verify_applied")
        if tc_state_is_clean(out):
            raise RuntimeError(
                f"congestion2: impairment did not stick on {v}: {out}")
    cluster.journal.log(action="applied", cls=cls_, victims=victims,
                        rate=args.rate, delay=args.delay,
                        jitter=args.jitter, loss=args.loss)


def remove_impairment(cluster, victims):
    for v in victims:
        cluster.run(v, teardown_cmds(cluster.iface)[0], action="remove",
                    check=False)
    cluster.journal.log(action="removed", victims=victims)


def teardown_all(cluster, nodes=None):
    nodes = nodes or list(NODES)
    ok = True
    for n in nodes:
        for c in teardown_cmds(cluster.iface):
            cluster.run(n, c, action="teardown", check=False)
    for n in nodes:
        rc, out = cluster.run(n, f"tc qdisc show dev {cluster.iface}",
                              action="verify_teardown", check=False)
        clean = (rc == 0) and tc_state_is_clean(out)
        print(f"  {n}: {'clean' if clean else 'DIRTY -> ' + out}")
        ok = ok and clean
    cluster.journal.log(action="teardown_verified", ok=ok)
    if not ok:
        raise RuntimeError("congestion2: teardown verification FAILED")
    return ok


def verify_all_clean(cluster):
    return teardown_all(cluster)  # teardown is idempotent + verifying


def c2_burst(cluster, victims, args, on_s):
    """One synchronized ON phase of organic cross traffic.

    Each sender targets its OWN iperf3 server port (base+idx) on the victim:
    one iperf3 server serves exactly ONE client at a time, so a shared port
    silently degrades the intended incast to a single active sender (observed
    as rcs=[0,1,1] on every burst of the 2026-07 matrix). Per-sender ports
    make the many-to-one incast real."""
    senders = [n for n in NODES if n not in victims]
    procs = []
    for v in victims:
        vip = NODES[v]["private"]
        for i, s in enumerate(senders):
            cmd = build_iperf_client_cmd(vip, port=args.iperf_port + i,
                                         streams=args.streams,
                                         duration_s=on_s,
                                         direction=args.direction)
            procs.append(cluster.run_bg(s, cmd, action="iperf_burst"))
            if not args.incast:
                break  # single sender unless incast (many-to-one)
    deadline = time.time() + on_s + 20
    rcs = []
    for p in procs:
        try:
            p.wait(timeout=max(1, deadline - time.time()))
        except subprocess.TimeoutExpired:
            # never let a hung client kill the whole congestion driver
            p.kill()
            try:
                p.wait(timeout=5)
            except Exception:
                pass
        rcs.append(p.returncode if p.returncode is not None else -1)
    ok = bool(rcs) and all(rc == 0 for rc in rcs)
    cluster.journal.log(action="iperf_burst_done", rcs=rcs, ok=ok)
    if not ok:
        print(f"congestion2: WARNING iperf3 burst rcs={rcs} "
              f"(a nonzero rc = that sender contributed NO traffic)",
              file=sys.stderr)


def ensure_iperf_servers(cluster, victims, args):
    """One iperf3 server per sender port on each victim (see c2_burst)."""
    n_ports = len([n for n in NODES if n not in victims]) if args.incast else 1
    for v in victims:
        rc, _ = cluster.run(v, "which iperf3", action="check_iperf", check=False)
        if rc != 0:
            raise RuntimeError(f"congestion2: iperf3 not installed on {v}; "
                               f"install it or drop c2 conditions")
        for i in range(n_ports):
            cluster.run(v, build_iperf_server_cmd(args.iperf_port + i),
                        action="iperf_server")


def run_mode(cluster, args):
    """Blocking driver for steady / squarewave / duty-cycled organic runs."""
    victims = args.victims.split(",")
    stop = {"flag": False}

    def _sig(_s, _f):
        stop["flag"] = True
    signal.signal(signal.SIGTERM, _sig)
    signal.signal(signal.SIGINT, _sig)

    t0 = time.time()
    cluster.journal.log(action="run_start", cls=args.cls, mode=args.mode,
                        victims=victims, duration=args.duration,
                        period=args.period, duty=args.duty, phase=args.phase,
                        rate=args.rate, delay=args.delay, jitter=args.jitter,
                        loss=args.loss, streams=args.streams,
                        direction=args.direction, incast=args.incast,
                        on=args.on, off=args.off)
    try:
        if args.cls == "c2":
            ensure_iperf_servers(cluster, victims, args)
            while not stop["flag"] and time.time() - t0 < args.duration:
                remaining = args.duration - (time.time() - t0)
                on_s = min(args.on, max(1, remaining))
                cluster.journal.log(action="phase_on", t_rel=time.time() - t0)
                c2_burst(cluster, victims, args, on_s)
                cluster.journal.log(action="phase_off", t_rel=time.time() - t0)
                _sleep_until(t0, min(args.duration,
                                     time.time() - t0 + args.off), stop)
        elif args.mode == "steady":
            apply_impairment(cluster, args.cls, victims, args)
            _sleep_until(t0, args.duration, stop)
        elif args.mode == "squarewave":
            sched = squarewave_schedule(args.duration, args.period,
                                        args.duty, args.phase)
            cluster.journal.log(action="schedule",
                                windows=[[round(s, 1), round(e, 1)]
                                         for s, e in sched])
            for (s, e) in sched:
                if stop["flag"]:
                    break
                _sleep_until(t0, s, stop)
                if stop["flag"]:
                    break
                cluster.journal.log(action="wave_on", t_rel=time.time() - t0)
                apply_impairment(cluster, args.cls, victims, args)
                _sleep_until(t0, e, stop)
                remove_impairment(cluster, victims)
                cluster.journal.log(action="wave_off", t_rel=time.time() - t0)
            _sleep_until(t0, args.duration, stop)
        else:
            raise ValueError(f"unknown mode {args.mode}")
    finally:
        teardown_all(cluster, victims + (["gnn1"] if args.cls == "c2" else []))
        cluster.journal.log(action="run_end", t_rel=time.time() - t0,
                            interrupted=stop["flag"])


def _sleep_until(t0, t_rel, stop):
    while not stop["flag"] and time.time() - t0 < t_rel:
        time.sleep(min(1.0, t_rel - (time.time() - t0)))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["apply", "run", "teardown", "verify",
                                       "status"])
    ap.add_argument("--cls", choices=["c1", "c2", "c4"], default="c1")
    ap.add_argument("--mode", choices=["steady", "squarewave"], default="steady")
    ap.add_argument("--victims", default="gnn4",
                    help="comma-separated victim node names (default gnn4)")
    ap.add_argument("--iface", default=DEFAULT_IFACE)
    ap.add_argument("--ssh-mode", choices=["auto", "peer", "public"],
                    default="auto")
    ap.add_argument("--journal", default="", help="JSONL journal path")
    # c1
    ap.add_argument("--rate", default=None, help="tbf rate, e.g. 200mbit")
    # c4
    ap.add_argument("--delay", type=float, default=None, help="netem delay ms")
    ap.add_argument("--jitter", type=float, default=None, help="netem jitter ms")
    ap.add_argument("--loss", type=float, default=None, help="gemodel loss %%")
    # c2
    ap.add_argument("--streams", type=int, default=8)
    ap.add_argument("--direction", choices=["egress", "ingress"],
                    default="egress")
    ap.add_argument("--incast", action="store_true",
                    help="all non-victim nodes send simultaneously (many-to-one)")
    ap.add_argument("--iperf-port", type=int, default=IPERF_PORT)
    ap.add_argument("--on", type=float, default=30.0, help="c2 ON seconds")
    ap.add_argument("--off", type=float, default=30.0, help="c2 OFF seconds")
    # run mode
    ap.add_argument("--duration", type=float, default=1800.0)
    ap.add_argument("--period", type=float, default=120.0)
    ap.add_argument("--duty", type=float, default=0.5)
    ap.add_argument("--phase", type=float, default=0.0)
    args = ap.parse_args()

    journal = Journal(args.journal or None)
    cluster = Cluster(ssh_mode=args.ssh_mode, journal=journal,
                      iface=args.iface)
    try:
        if args.action == "apply":
            apply_impairment(cluster, args.cls, args.victims.split(","), args)
        elif args.action == "run":
            run_mode(cluster, args)
        elif args.action == "teardown":
            teardown_all(cluster)
        elif args.action == "verify":
            verify_all_clean(cluster)
        elif args.action == "status":
            for n in NODES:
                rc, out = cluster.run(n, f"tc qdisc show dev {args.iface}",
                                      action="status", check=False)
                print(f"--- {n} (rc={rc}) ---\n{out}")
    finally:
        journal.close()


if __name__ == "__main__":
    main()
