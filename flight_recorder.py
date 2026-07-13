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
    def __init__(self, out_path, interval_s=1.0, iface="eno1", gpu_index=None):
        super().__init__(daemon=True, name="flight-recorder")
        self.out_path = out_path
        self.interval_s = float(interval_s)
        self.iface = iface
        self._stop_evt = threading.Event()

        nic_dir = f"/sys/class/net/{iface}/statistics"
        self._nic = nic_dir if os.path.isdir(nic_dir) else None

        self._rapl = [p for p in
                      sorted(glob.glob("/sys/class/powercap/intel-rapl*/energy_uj"))
                      if os.access(p, os.R_OK)]

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

    @staticmethod
    def _read_int(path):
        try:
            with open(path) as f:
                return int(f.read().strip())
        except Exception:
            return None

    def _sample(self):
        rec = {"t": time.time()}
        if self._nic:
            rec["rx"] = self._read_int(os.path.join(self._nic, "rx_bytes"))
            rec["tx"] = self._read_int(os.path.join(self._nic, "tx_bytes"))
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
        meta = {"meta": True, "t": time.time(), "iface": self.iface,
                "interval_s": self.interval_s,
                "nic": bool(self._nic), "rapl_domains": self._rapl,
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
        self.join(timeout=timeout)
        if self._nvml:
            try:
                self._nvml.nvmlShutdown()
            except Exception:
                pass
