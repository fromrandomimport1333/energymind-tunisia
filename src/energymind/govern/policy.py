"""Goal 3 - Version 1 escalation engine (docs/goal3_governance).

The route of a case is the maximum of the floors set by rules R0-R9, so a rule can only raise it.
Confidence bands, risk, agreement states and action locks follow the report; thresholds are the
report's prototype values, not STEG values. The request-more-data loop, the operator console and
the Version 2 learned rejector are design only.
"""
from dataclasses import dataclass, field

LEVELS = ["G0", "G1", "G2a", "G2b", "G3"]
CLASSES = ["NORMAL", "GENUINE_EVENT", "METER_FAULT", "SUSPICIOUS_DATA"]
ORD = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]

TAU_HIGH = {"GENUINE_EVENT": 0.85}          # default 0.90
TAU_LOW, MIN_MARGIN, RHO_PLAUS, RHO_PARTIAL = 0.60, 0.30, 0.05, 0.20

# missed-event risk M[criticality][severity]
RISK_MATRIX = {
    "LOW":      ["LOW", "LOW", "MEDIUM", "HIGH"],
    "MEDIUM":   ["LOW", "MEDIUM", "MEDIUM", "HIGH"],
    "HIGH":     ["MEDIUM", "MEDIUM", "HIGH", "CRITICAL"],
    "CRITICAL": ["MEDIUM", "HIGH", "CRITICAL", "CRITICAL"],
}
BASE_RISK = {"informational": "LOW", "data": "LOW", "operational": "MEDIUM",
             "physical": "HIGH", "customer": "HIGH", "accusatory": "HIGH"}
HARD_KINDS = {"physical", "customer", "accusatory"}
CRITICAL_PAIRS = [{"GENUINE_EVENT", "METER_FAULT"}, {"GENUINE_EVENT", "SUSPICIOUS_DATA"},
                  {"GENUINE_EVENT", "NORMAL"}]

# dataset gt_expected_response -> nearest governance level (missed / false escalations)
REFERENCE = {"auto_log": "G0", "auto_correct": "G1", "notify_operator": "G2a", "human_decision": "G2b"}


@dataclass
class Action:
    name: str
    kind: str                     # informational | data | operational | physical | customer | accusatory
    irreversible: bool = False
    whitelisted: bool = False     # may run automatically (G1)


@dataclass
class Case:
    probs: dict                                   # p(c | x) for the four classes (Goal 2)
    criticality: str = "LOW"                      # asset registry (configs/assets.yaml)
    severity: dict = field(default_factory=dict)  # severity of the event under each hypothesis
    actions: list = field(default_factory=list)
    calibrated: frozenset = frozenset({"NORMAL", "GENUINE_EVENT", "METER_FAULT"})  # N1: SUSPICIOUS_DATA is not
    ood: bool = False
    data_quality_good: bool = True
    inputs_valid: bool = True
    no_data_intervals: int = 0                    # > 0: communication loss (NO_DATA)
    fault_kind: str = ""                          # METER_FAULT: billing | voltage_offset | clock_shift | stuck | corruption
    event_kind: str = ""                          # GENUINE_EVENT: scheduled | equipment | other
    duration_h: float = 0.0
    recurrence: int = 0                           # automatic actions on this asset in the window
    blind: tuple = None                           # operator's blind (class, "LOW"|"MID"|"HIGH")


@dataclass
class Decision:
    route: str
    band: str
    risk: str
    agreement: str
    locked: list
    two_person: bool
    flags: set
    fired: list


def up(level, n=1):
    return ORD[min(ORD.index(level) + n, len(ORD) - 1)]


def diagnosis(case):
    return max(case.probs, key=case.probs.get)


def band(case):
    c = diagnosis(case)
    p = case.probs[c]
    b = "HIGH" if p >= TAU_HIGH.get(c, 0.90) else "LOW" if p < TAU_LOW else "MEDIUM"
    if b == "HIGH" and (case.ood or not case.data_quality_good or c not in case.calibrated):
        b = "MEDIUM"
    return b


def margin(case):
    p = sorted(case.probs.values(), reverse=True)
    return p[0] - p[1]


def action_risk(a, criticality):
    r = BASE_RISK[a.kind]
    if a.irreversible:
        r = up(r)
    if a.kind in {"physical", "customer"}:
        r = max(r, criticality, key=ORD.index)
    return r


def case_risk(case):
    plausible = [c for c, p in case.probs.items() if p >= RHO_PLAUS]
    sev = max((case.severity.get(c, "LOW") for c in plausible), key=ORD.index)
    miss = RISK_MATRIX[case.criticality][ORD.index(sev)]
    return max([miss] + [action_risk(a, case.criticality) for a in case.actions], key=ORD.index)


def agreement(case):
    if case.blind is None:
        return "NOT_ASSESSED"
    d_a, (d_h, lvl) = diagnosis(case), case.blind
    b, h = ["LOW", "MEDIUM", "HIGH"].index(band(case)), ["LOW", "MID", "HIGH"].index(lvl)
    if d_h == d_a:
        return "AGREEMENT" if min(b, h) >= 1 else "WEAK_AGREEMENT"
    if case.probs.get(d_h, 0.0) >= RHO_PARTIAL:
        return "PARTIAL"
    gamma = (b == h == 2) or ({d_a, d_h} in CRITICAL_PAIRS and b >= 1 and h >= 1)
    return "STRONG" if gamma else "PARTIAL"


def route(case, with_agreement=True):
    floors, fired, flags, locked = [], [], set(), []

    def floor(rule, level):
        floors.append(level); fired.append(rule)

    c, b, risk = diagnosis(case), band(case), case_risk(case)
    # R0 input validity (NO_DATA: short gaps are filled automatically)
    if not case.inputs_valid:
        floor("R0", "G2b")
    if case.no_data_intervals:
        floor("R0", "G1" if case.no_data_intervals <= 12 else "G2b")
    # R1 hard actions are always locked
    for a in case.actions:
        if a.kind in HARD_KINDS or a.irreversible:
            locked.append(a.name); fired.append("R1")
    # R2 consequence: critical risk, or a critical asset with a non-normal diagnosis
    if risk == "CRITICAL" or (case.criticality == "CRITICAL" and c != "NORMAL"):
        floor("R2", "G2b")
        locked += [a.name for a in case.actions if not a.whitelisted and a.name not in locked]
    elif risk == "HIGH":
        floor("R2b", "G2b")
    # R3 agreement (pass 2 only)
    agree = agreement(case) if with_agreement else "NOT_ASSESSED"
    if agree in {"WEAK_AGREEMENT", "PARTIAL", "STRONG"}:
        floor("R3", "G2b")
        flags |= {"more_data"} if agree != "PARTIAL" else set()
        if agree == "STRONG":
            flags |= {"second_review", "freeze"}
    # R4 familiarity
    if case.ood:
        floor("R4", "G2b"); flags.add("more_data")
    # R5 confidence
    if b == "LOW":
        floor("R5", "G2b"); flags.add("more_data")
    elif b == "MEDIUM" or margin(case) < MIN_MARGIN:
        floor("R5b", "G2a")
    # R6 diagnosis defaults
    if c == "NORMAL":
        floor("R6", "G0")
    elif c == "METER_FAULT":
        if case.fault_kind == "billing" or case.duration_h >= 24:
            floor("R6", "G2b")
        elif case.fault_kind in {"voltage_offset", "clock_shift"}:
            floor("R6", "G2a")
        else:
            floor("R6", "G1")
    elif c == "GENUINE_EVENT":
        sev = case.severity.get("GENUINE_EVENT", "LOW")
        if case.event_kind == "equipment" and ORD.index(sev) >= 1:
            floor("R6", "G2b")
        elif case.event_kind == "scheduled":
            floor("R6", "G0")
        else:
            floor("R6", "G2a")
    else:                                          # SUSPICIOUS_DATA
        floor("R6", "G2b")
    # R7 recurrence
    if case.recurrence >= 3:
        floor("R7", "G2b")
    # R9 consistency: locked action -> G3; any other non-whitelisted action -> G2a
    if locked:
        floor("R9", "G3")
    elif any(not a.whitelisted for a in case.actions):
        floor("R9", "G2a")

    r = max(floors, key=LEVELS.index) if floors else "G0"
    return Decision(route=r, band=b, risk=risk, agreement=agree, locked=sorted(set(locked)),
                    two_person=bool(locked) and risk == "CRITICAL", flags=flags, fired=fired)


def compare_to_reference(route_level, expected_response):
    """-1 = below reference (missed escalation), 0 = matches, +1 = above (false escalation)."""
    if expected_response == "none":
        return 0
    a, b = LEVELS.index(route_level), LEVELS.index(REFERENCE[expected_response])
    return (a > b) - (a < b)
