"""Rule cascade: evidence -> verdict + sub-type + rule id (Goal 2, Fig. 1, with the review fixes).

Order matters: the first rule that matches is reported.
  R0 supply lost          -> GENUINE_EVENT (outage / local supply loss)
  R1 missing readings     -> NO_DATA (communication loss)
  R2 frozen value         -> METER_FAULT (stuck sensor)
  R3 V-I-P inconsistent   -> SUSPICIOUS_DATA (energy below V*I) / METER_FAULT (impossible value)
  R4 neighbours agree     -> GENUINE_EVENT (collective)
  R5 drift / offset       -> METER_FAULT (CUSUM on neighbour ratio)
  R5b consistent drop + feeder balance alarm -> SUSPICIOUS_DATA (possible bypass)
  R6 otherwise            -> GENUINE_EVENT (single-house load)
"""
from dataclasses import dataclass


@dataclass
class Verdict:
    verdict: str
    subtype: str
    rule: str
    reason: str


def decide(e, th):
    """e: dict of evidence values for one flagged reading; th: thresholds (configs/thresholds.yaml)."""
    if e["v_zero"]:
        sub = "outage" if e["s"] >= th["s"] else "local_supply_loss"
        return Verdict("GENUINE_EVENT", sub, "R0",
                       f"Voltage is 0 V on {e['n_aff']} of {e['n_tot']} meters: supply was lost.")
    if e["g"] >= th["g"]:
        return Verdict("NO_DATA", "comm_loss", "R1", f"{e['g']:.0%} of recent readings are missing.")
    if e["phi"] <= th["phi"]:
        return Verdict("METER_FAULT", "stuck_sensor", "R2", "The reading has stopped varying.")
    if e["impossible"]:
        return Verdict("METER_FAULT", "corruption", "R3", "The value is physically impossible.")
    if e["rho"] is not None and abs(e["rho"] - 1) > th["rho"]:
        return Verdict("SUSPICIOUS_DATA", "data_only_fdi", "R3",
                       f"Reported energy is {e['rho']:.2f}x what voltage and current imply.")
    if e["s"] >= th["s"]:
        return Verdict("GENUINE_EVENT", "collective", "R4",
                       f"{e['n_aff']} of {e['n_tot']} meters changed together and physics holds.")
    if e["cusum_alarm"]:
        return Verdict("METER_FAULT", "drift_or_bias", "R5", "This meter is drifting away from its neighbours.")
    if e["drop"] >= th["drop"] and e["balance_alarm"]:
        return Verdict("SUSPICIOUS_DATA", "possible_bypass", "R5b",
                       f"Reports {e['drop']:.0%} less than usual while the feeder is losing energy.")
    return Verdict("GENUINE_EVENT", "single_house_load", "R6", "Short, isolated and physically consistent.")
