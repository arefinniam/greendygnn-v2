"""1 Hz node-local flight recorder (RESEARCH_PLAN_v2 item 3).

Records a JSONL time series per rank: NIC byte counters, raw /proc/stat CPU
line, RAPL energy counters, GPU power/util via NVML. Purpose: ground-truth
*achieved* congestion exposure (not just commanded tc state), per-phase energy
attribution, and real bandwidth timelines for simulator replay.

Design rules:
  - raw counters only; all derivation (rates, utilization) happens offline;
  - every source degrades gracefully (absent sysfs path / NVML → omitted);
  - the first line is a meta record incl. `chronyc tracking` output so
    cross-node timestamps can be aligned offline;
  - overhead budget ~1 sample/s of a few sysfs reads — negligible, but
    measure it anyway (plan: instrumentation-overhead A/B via the
    --flight_recorder flag being opt-in).
"""

import glob
import json
import os
import subprocess
import threading
import time


class FlightRecorder(threading.Thread):
    def __init__(self, out_path, interval_s=1.0, iface="eno1", gpu_index=None,
                 rank=None):
        super().__init__(daemon=True, name="flight-recorder")
        self.out_path = out_path
        self.interval_s = float(interval_s)
        self.iface = iface
        self.rank = rank
        self._stop_evt = threading.Event()

        # Node-source ownership: with >1 rank per node, only ONE recorder may
        # record node-level sources (NIC/CPU/RAPL) or they would be double
        # counted downstream. First recorder to claim the per-node lockfile
        # wins; the others record GPU only. Stale locks (dead pid) are taken
        # over.
        self._lock_path = os.path.join(
            "/tmp", f"flight_recorder_node.{iface}.lock")
        self.node_sources = self._claim_node_sources()

        nic_dir = f"/sys/class/net/{iface}/statistics"
        self._nic = nic_dir if (self.node_sources and
                                os.path.isdir(nic_dir)) else None

        self._rapl = [p for p in
                      sorted(glob.glob("/sys/class/powercap/intel-rapl*/energy_uj"))
                      if os.access(p, os.R_OK)] if self.node_sources else []
        # counter wrap bound per domain, for offline rollover correction
        self._rapl_max = [self._read_int(
            os.path.join(os.path.dirname(p), "max_energy_range_uj"))
            for p in self._rapl]

        self._nvml = None
        self._gpu_handles = []
        try:
            import pynvml
            pynvml.nvmlInit()
            n = pynvml.nvmlDeviceGetCount()
            idxs = range(n) if gpu_index is None else [int(gpu_index)]
            self._gpu_handles = [pynvml.nvmlDeviceGetHandleByIndex(i)
                                 for i in idxs]
            self._nvml = pynvml
        except Exception:
            self._nvml = None

    def _claim_node_sources(self):
        try:
            fd = os.open(self._lock_path,
                         os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            return True
        except FileExistsError:
            try:
                holder = int(open(self._lock_path).read().strip())
                if holder != os.getpid() and not os.path.exists(
                        f"/proc/{holder}"):
                    os.unlink(self._lock_path)      # stale — take over
                    return self._claim_node_sources()
            except Exception:
                pass
            return False
        except Exception:
            return True    # lockfile unavailable: record rather than lose data

    def _release_node_sources(self):
        if not self.node_sources:
            return
        try:
            if int(open(self._lock_path).read().strip()) == os.getpid():
                os.unlink(self._lock_path)
        except Exception:
            pass

    @staticmethod
    def _read_int(path):
        try:
            with open(path) as f:
                return int(f.read().strip())
        except Exception:
            return None

    def _sample(self):
        rec = {"t": time.time(), "tm": time.monotonic_ns()}
        if self._nic:
            rec["rx"] = self._read_int(os.path.join(self._nic, "rx_bytes"))
            rec["tx"] = self._read_int(os.path.join(self._nic, "tx_bytes"))
        if self.node_sources:
            try:
                with open("/proc/stat") as f:
                    rec["cpu"] = f.readline().strip()
            except Exception:
                pass
        if self._rapl:
            rec["rapl"] = [self._read_int(p) for p in self._rapl]
        if self._nvml:
            gpus = []
            for h in self._gpu_handles:
                try:
                    gpus.append({
                        "p_mw": self._nvml.nvmlDeviceGetPowerUsage(h),
                        "util": self._nvml.nvmlDeviceGetUtilizationRates(h).gpu,
                        "sm_mhz": self._nvml.nvmlDeviceGetClockInfo(
                            h, self._nvml.NVML_CLOCK_SM),
                    })
                except Exception:
                    gpus.append(None)
            rec["gpu"] = gpus
        return rec

    def _meta(self):
        meta = {"meta": True, "t": time.time(), "tm": time.monotonic_ns(),
                "iface": self.iface, "interval_s": self.interval_s,
                "nic": bool(self._nic), "rapl_domains": self._rapl,
                "rapl_max_range_uj": self._rapl_max,
                "node_sources": self.node_sources,
                "rank": self.rank, "pid": os.getpid(),
                "gpus": len(self._gpu_handles), "host": os.uname().nodename}
        try:
            meta["chrony"] = subprocess.run(
                ["chronyc", "tracking"], capture_output=True, text=True,
                timeout=5).stdout
        except Exception:
            meta["chrony"] = None
        return meta

    def run(self):
        try:
            with open(self.out_path, "w") as f:
                f.write(json.dumps(self._meta()) + "\n")
                while not self._stop_evt.is_set():
                    f.write(json.dumps(self._sample()) + "\n")
                    f.flush()
                    self._stop_evt.wait(self.interval_s)
        except Exception as e:  # never take the trainer down
            print(f"[FlightRecorder] died: {type(e).__name__}: {e}")

    def stop(self, timeout=5.0):
        self._stop_evt.set()
        if self.ident is not None:      # join only if the thread ever started
            self.join(timeout=timeout)
        self._release_node_sources()
        if self._nvml:
            try:
                self._nvml.nvmlShutdown()
            except Exception:
                pass
