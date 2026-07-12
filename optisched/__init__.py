"""OptiSched-GNN: trace-aware optimal rebuild scheduling.

Offline core (Phase-0 gate + system):
  calibration.CostModel        calibrated cost model (A3) + structural coeffs
                               (eps_init, kappa, d, C_max) for the floor (Arch §2-3)
  trace.Trace / SyntheticTrace  deterministic per-batch remote access trace (A1)
  interval_cost.IntervalCosts   sigma-independent per-interval residual misses;
                               weighted_template = Theorem-B hot-set selection
  dp_solver                     Thm C global-window DP, Thm E oracle-uniform sweep,
                               Thm D Option-B delta DP, Thm G owner-decoupled DP
  floor                         Thm A communication floor L_0, Thm F near-floor
                               decomposition (rho, gamma_R, phi, eta), Lemma D
                               lifetime-fit, working-set drift beta (Arch §15)
  regime.RegimeLibrary          per-regime DP schedules
  regime.FixedShareController    Thm H no-regret online controller (replaces DQN),
  regime.SimpleFixedShare        with explicit switching cost (Arch §16)
  gate                          Phase-0 gate experiment (timescale-swept, leak-guarded)

Everything in this package is pure numpy/torch and runs offline; nothing here
adds work to the existing training loop unless explicitly imported by a trainer.
"""

from .calibration import CostModel
from .trace import Trace, SyntheticTrace

__all__ = ["CostModel", "Trace", "SyntheticTrace"]
