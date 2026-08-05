"""
Layer 2 - Storage
=================
Represents the storage system as a set of physical and operational
parameters, plus the state-of-charge (SOC) bookkeeping that follows
mechanically from them.

Physical parameters:
  - a power rating                    (P_max)
  - an energy capacity                (E_cap)
  - charge and discharge efficiencies (η_c, η_d)
  - SOC bounds                        (SOC_min, SOC_max)

Operational parameters - per-MWh-discharged costs of running the unit,
one economic and one environmental:
  - a cycle (degradation) cost        (cycle_cost_eur_mwh)
  - an embodied emission cost         (embodied_emission_kg_mwh)

This class is technology-agnostic: any storage technology can be
represented here, provided it can be characterised by these seven
numbers. The paper's worked example is a specific lithium-ion battery
(Table 3, "Full overview of battery storage system specifications") - 
that is one instantiation of this class, not a constraint of it. Nothing
here assumes battery chemistry; the same seven parameters could equally
describe a pumped-hydro plant, a flow battery, or compressed-air storage.

This layer contains no decision logic. Its methods only answer two
mechanical questions - "what charge/discharge power is physically
feasible from this SOC" and "what SOC results from taking this action" - 
so that Layer 3 (algorithms.py) never has to re-derive the physics and
can never propose an infeasible action. Deciding *which* action to take
is entirely Layer 3's responsibility; this layer has no say in it.

Paper reference: Section 3.3
"""

import math
from dataclasses import dataclass


@dataclass
class StorageUnit:
    """Physical and operational parameters of a grid-scale storage system.

    Technology-agnostic: nothing here is specific to batteries. The first
    five attributes are the physical parameters; the paper's worked
    example (a lithium-ion battery, Table 3) is one instantiation of them,
    not a constraint of the class. The last two are operational
    parameters - fixed per-MWh-discharged cost proxies of the modelled
    hardware (e.g. a specific battery product's degradation warranty and
    its Environmental Product Declaration), not a per-run configuration
    choice - so both live on the unit itself, exactly like p_max_mw or
    eta_c do.

    Attributes
    ----------
    p_max_mw : float
        Maximum charge or discharge power [MW].
        Constraint: 0 ≤ P_c ≤ P_max, 0 ≤ P_d ≤ P_max  (Section 3.3).
    e_cap_mwh : float
        Usable energy capacity [MWh].
    eta_c : float
        Charge efficiency η_c  (fraction in (0, 1]).
    eta_d : float
        Discharge efficiency η_d (fraction in (0, 1]).
    soc_min : float
        Minimum state-of-charge as a fraction of capacity (default 0).
    soc_max : float
        Maximum state-of-charge as a fraction of capacity (default 1).
    cycle_cost_eur_mwh : float
        Economic degradation cost per MWh discharged [EUR / MWh].
        Used for the profit objective (Section 3.4, Eq. 3.4.1), where the
        DP signal is price [EUR/MWh].
    embodied_emission_kg_mwh : float
        Embodied greenhouse-gas cost per MWh discharged [kg CO2eq / MWh],
        amortising the unit's cradle-to-gate embodied emissions
        (manufacturing, distribution, installation, end-of-life - 
        excluding the use stage) over its warranted cycle life. Used for
        the emission objective (Section 3.4, Eq. 3.4.1), where the DP
        signal is MEI [kg CO2/kWh_e] - see algorithms.dp's `discharge_cost`
        parameter for the kg/MWh → kg/kWh scaling. Set to 0 to ignore
        embodied emissions.
    """

    p_max_mw:                 float
    e_cap_mwh:                float
    eta_c:                    float
    eta_d:                    float
    soc_min:                  float = 0.0
    soc_max:                  float = 1.0
    cycle_cost_eur_mwh:       float = 0.0
    embodied_emission_kg_mwh: float = 0.0

    # ------------------------------------------------------------------
    # Constructor helper
    # ------------------------------------------------------------------

    @classmethod
    def from_roundtrip_efficiency(
        cls,
        p_max_mw: float,
        e_cap_mwh: float,
        roundtrip_efficiency: float,
        soc_min: float = 0.0,
        soc_max: float = 1.0,
        cycle_cost_eur_mwh: float = 0.0,
        embodied_emission_kg_mwh: float = 0.0,
    ) -> "StorageUnit":
        """Create a unit with symmetric charge/discharge efficiencies.

        The round-trip efficiency η_RT is split symmetrically (Section 3.3):
            η_c = η_d = √η_RT

        Example (illustrative round numbers, not the paper's case study - 
        see run.py's UNIT for the actual battery specification, Table 3):
            StorageUnit.from_roundtrip_efficiency(
                p_max_mw=1.0, e_cap_mwh=1.0,
                roundtrip_efficiency=0.85,
                cycle_cost_eur_mwh=30.0,
            )
            → η_c = η_d ≈ 0.922
        """
        if not (0 < roundtrip_efficiency <= 1):
            raise ValueError("roundtrip_efficiency must be in (0, 1].")
        eta = math.sqrt(roundtrip_efficiency)
        return cls(
            p_max_mw=p_max_mw,
            e_cap_mwh=e_cap_mwh,
            eta_c=eta,
            eta_d=eta,
            soc_min=soc_min,
            soc_max=soc_max,
            cycle_cost_eur_mwh=cycle_cost_eur_mwh,
            embodied_emission_kg_mwh=embodied_emission_kg_mwh,
        )

    # ------------------------------------------------------------------
    # Derived SOC limits in MWh
    # ------------------------------------------------------------------

    @property
    def soc_min_mwh(self) -> float:
        """Minimum SOC in MWh."""
        return self.soc_min * self.e_cap_mwh

    @property
    def soc_max_mwh(self) -> float:
        """Maximum SOC in MWh."""
        return self.soc_max * self.e_cap_mwh

    # ------------------------------------------------------------------
    # Feasible power limits at a given SOC
    # ------------------------------------------------------------------

    def max_charge_mw(self, soc_mwh: float) -> float:
        """Maximum charge power [MW] that keeps SOC ≤ SOC_max.

        Derived from the charging equation (Section 3.3, Eq. 1):
            SOC_{t+1} = SOC_t + P_c × η_c   ≤ SOC_max
            → P_c ≤ (SOC_max - SOC_t) / η_c
        """
        available = (self.soc_max_mwh - soc_mwh) / self.eta_c
        return min(self.p_max_mw, max(0.0, available))

    def max_discharge_mw(self, soc_mwh: float) -> float:
        """Maximum discharge power [MW] that keeps SOC ≥ SOC_min.

        Derived from the discharging equation (Section 3.3, Eq. 2):
            SOC_{t+1} = SOC_t - P_d / η_d   ≥ SOC_min
            → P_d ≤ (SOC_t - SOC_min) × η_d
        """
        available = (soc_mwh - self.soc_min_mwh) * self.eta_d
        return min(self.p_max_mw, max(0.0, available))

    # ------------------------------------------------------------------
    # SOC transition equations  (Section 3.3)
    # ------------------------------------------------------------------

    def soc_after_charge(self, power_mw: float, soc_mwh: float) -> float:
        """New SOC after charging for one hour at power_mw.

        Equation 3.3.1 (paper):
            SOC_{t+1} = SOC_t + P_c × η_c × Δt    (Δt = 1 h)
        """
        return soc_mwh + power_mw * self.eta_c

    def soc_after_discharge(self, power_mw: float, soc_mwh: float) -> float:
        """New SOC after discharging for one hour at power_mw.

        Equation 3.3.2 (paper):
            SOC_{t+1} = SOC_t - (P_d / η_d) × Δt  (Δt = 1 h)
        """
        return soc_mwh - power_mw / self.eta_d
