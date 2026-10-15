"""Goal 3 engine: the six N1 reference scenarios and five safety invariants.

Each invariant is checked on 400 random cases (fixed seed), as stated in the Goal 3 report.
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from energymind.govern.policy import (CLASSES, LEVELS, ORD, Action, Case,  # noqa: E402
                                      compare_to_reference, diagnosis, route)

N = 400
ge = LEVELS.index


def probs(top, p, rest=None):
    others = [c for c in CLASSES if c != top]
    rest = rest or {}
    left = 1 - p - sum(rest.values())
    out = {c: rest.get(c, left / (len(others) - len(rest))) for c in others}
    out[top] = p
    return out


# ------------------------------------------------------------- reference scenarios (N1 events)
QUARANTINE = Action("quarantine data", "data", whitelisted=True)
REREAD = Action("re-read meter", "informational", whitelisted=True)
TICKET = Action("open ticket", "informational", whitelisted=True)


def test_s1_stuck_meter_auto_then_review():          # N1-F1, 12-14 Feb
    case = Case(probs("METER_FAULT", 0.95), severity={"METER_FAULT": "MEDIUM"},
                actions=[QUARANTINE, REREAD, TICKET], fault_kind="stuck", duration_h=6)
    assert route(case).route == "G1"
    case.duration_h = 30                              # the freeze outlasts 24 h
    assert route(case).route == "G2b"


def test_s2_sag_on_critical_feeder_needs_two_people():   # N1-F3, 27 Jun
    transfer = Action("transfer to backup feeder", "physical")
    case = Case(probs("GENUINE_EVENT", 0.98), criticality="CRITICAL",
                severity={"GENUINE_EVENT": "HIGH"}, actions=[transfer, REREAD])
    d = route(case)
    assert d.route == "G3" and "transfer to backup feeder" in d.locked and d.two_person
    assert "re-read meter" not in d.locked           # safe actions still run


def test_s3_gain_drift_billing_reviewed_even_if_operator_agrees():   # N1-F2, from 19 Apr
    case = Case(probs("METER_FAULT", 0.92), severity={"METER_FAULT": "MEDIUM"},
                fault_kind="billing", blind=("METER_FAULT", "HIGH"))
    d = route(case)
    assert d.agreement == "AGREEMENT" and d.route == "G2b"


def test_s4_demand_surge_strong_disagreement():       # N1-F2, 23 Jun
    case = Case(probs("METER_FAULT", 0.93, {"GENUINE_EVENT": 0.05}), criticality="MEDIUM",
                severity={"GENUINE_EVENT": "HIGH", "METER_FAULT": "LOW"},
                blind=("GENUINE_EVENT", "HIGH"))
    d = route(case)
    assert d.agreement == "STRONG" and ge(d.route) >= ge("G2b")
    assert {"second_review", "more_data", "freeze"} <= d.flags


def test_s5_unfamiliar_pattern_reviewed_with_more_data():   # clock shift, N1-F1, 23 Feb
    case = Case(probs("SUSPICIOUS_DATA", 0.46, {"GENUINE_EVENT": 0.30}), ood=True,
                blind=("GENUINE_EVENT", "LOW"))
    d = route(case)
    assert d.band == "LOW" and d.route == "G2b" and "more_data" in d.flags and not d.locked


def test_s6_fraud_never_automatic():                  # partial bypass, N1-F2, 4 Feb
    declare = Action("declare fraud and disconnect", "accusatory", irreversible=True)
    case = Case(probs("SUSPICIOUS_DATA", 0.94), severity={"SUSPICIOUS_DATA": "MEDIUM"},
                actions=[declare], blind=("SUSPICIOUS_DATA", "HIGH"))
    d = route(case)
    assert d.band == "MEDIUM"                         # not calibrated on N1 -> capped
    assert d.route == "G3" and d.locked and d.two_person


# ------------------------------------------------------------- safety invariants, 400 random cases each
KINDS = ["informational", "data", "operational", "physical", "customer", "accusatory"]


def random_case(rng):
    top = rng.choice(CLASSES)
    p = {c: rng.random() for c in CLASSES}
    p[top] += rng.uniform(0, 3)
    s = sum(p.values())
    acts = [Action(f"a{i}", rng.choice(KINDS), irreversible=rng.random() < 0.2,
                   whitelisted=rng.random() < 0.5) for i in range(rng.randint(0, 3))]
    blind = None if rng.random() < 0.4 else (rng.choice(CLASSES), rng.choice(["LOW", "MID", "HIGH"]))
    return Case({c: v / s for c, v in p.items()}, criticality=rng.choice(ORD),
                severity={c: rng.choice(ORD) for c in CLASSES}, actions=acts,
                ood=rng.random() < 0.1, data_quality_good=rng.random() < 0.9,
                no_data_intervals=rng.choice([0, 0, 0, 6, 30]),
                fault_kind=rng.choice(["", "billing", "voltage_offset", "stuck"]),
                event_kind=rng.choice(["", "scheduled", "equipment"]),
                duration_h=rng.uniform(0, 48), recurrence=rng.randint(0, 4), blind=blind)


def cases(seed):
    rng = random.Random(seed)
    return [random_case(rng) for _ in range(N)]


def test_hard_actions_always_locked_and_g3():
    for case in cases(1):
        d = route(case)
        hard = [a.name for a in case.actions if a.kind in {"physical", "customer", "accusatory"} or a.irreversible]
        assert set(hard) <= set(d.locked)
        if hard:
            assert d.route == "G3"


def test_critical_asset_never_below_review():
    for case in cases(2):
        if case.criticality == "CRITICAL" and diagnosis(case) != "NORMAL":
            assert ge(route(case).route) >= ge("G2b")


def test_confidence_cannot_unlock_what_is_at_stake():
    """The report's proposition: for a fixed non-NORMAL diagnosis, whatever the AI confidence,
    (a) hard actions stay locked and the route is G3; (b) on a CRITICAL asset every proposed
    non-whitelisted action stays locked. (Confidence may still lower the missed-event risk:
    a more confident diagnosis can rule out a severe alternative, as intended.)"""
    for case in cases(3):
        c = diagnosis(case)
        if c == "NORMAL":
            continue
        hard = {a.name for a in case.actions if a.kind in {"physical", "customer", "accusatory"} or a.irreversible}
        soft = {a.name for a in case.actions if not a.whitelisted}
        for p in (0.30, 0.60, 0.90, 0.99):
            case.probs = {k: (p if k == c else (1 - p) / 3) for k in case.probs}
            d = route(case)
            assert hard <= set(d.locked) and (d.route == "G3" or not hard)
            if case.criticality == "CRITICAL":
                assert soft <= set(d.locked) and ge(d.route) >= ge("G2b")


def test_second_pass_never_lowers_the_route():
    for case in cases(4):
        assert ge(route(case, with_agreement=True).route) >= ge(route(case, with_agreement=False).route)


def test_nothing_unsafe_runs_automatically():
    for case in cases(5):
        d = route(case)
        if d.route in {"G0", "G1"}:
            assert not d.locked and all(a.whitelisted for a in case.actions)
        if diagnosis(case) == "SUSPICIOUS_DATA":
            assert ge(d.route) >= ge("G2b")


def test_reference_policy():
    assert compare_to_reference("G1", "human_decision") == -1     # missed escalation
    assert compare_to_reference("G2b", "human_decision") == 0
    assert compare_to_reference("G2b", "auto_log") == 1           # false escalation
