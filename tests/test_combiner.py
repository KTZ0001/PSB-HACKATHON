"""Phase 3 tests: score combiner, tiered actions, templated explanation."""
from src.scoring.combiner import (
    ScoreCombiner, CombinerConfig,
    ACTION_ALLOW, ACTION_STEP_UP, ACTION_COOLING_OFF, ACTION_BLOCK,
    TYPE_LEGIT, TYPE_ATO, TYPE_MULE, TYPE_SE,
)
from src.explain.templated import TemplatedExplanationGenerator

combiner = ScoreCombiner()


def test_all_clean_is_high_trust_allow():
    r = combiner.combine(anomaly_risk=10, xgb_fraud_proba=0.02, xgb_mule_proba=0.0,
                         mule_graph_risk=0.0, social_engineering_score=0.0)
    assert r.trust_score >= 70
    assert r.predicted_type == TYPE_LEGIT
    assert r.recommended_action == ACTION_ALLOW


def test_clean_device_does_not_dilute_social_engineering():
    """The thesis test: low behavioural/mule risk must NOT rescue a high SE score."""
    r = combiner.combine(anomaly_risk=15, xgb_fraud_proba=0.05, xgb_mule_proba=0.0,
                         mule_graph_risk=0.0, social_engineering_score=74.0)
    assert r.predicted_type == TYPE_SE
    # noisy-OR keeps risk high -> trust well below 'allow'
    assert r.trust_score < 70
    assert r.recommended_action in (ACTION_STEP_UP, ACTION_COOLING_OFF, ACTION_BLOCK)


def test_high_behavioural_is_account_takeover():
    r = combiner.combine(anomaly_risk=60, xgb_fraud_proba=0.85, xgb_mule_proba=0.05,
                         mule_graph_risk=0.0, social_engineering_score=0.0)
    assert r.predicted_type == TYPE_ATO
    assert r.trust_score < 35


def test_high_mule_graph_is_mule_network():
    r = combiner.combine(anomaly_risk=20, xgb_fraud_proba=0.2, xgb_mule_proba=0.1,
                         mule_graph_risk=1.0, social_engineering_score=10.0)
    assert r.predicted_type == TYPE_MULE


def test_tiered_action_thresholds():
    c = ScoreCombiner(CombinerConfig())
    assert c.decide_action(90)[0] == ACTION_ALLOW
    assert c.decide_action(50)[0] == ACTION_STEP_UP
    assert c.decide_action(20)[0] == ACTION_COOLING_OFF
    assert c.decide_action(5)[0] == ACTION_BLOCK
    # cooling-off carries a configured delay
    assert c.decide_action(20)[1]["delay_minutes"] == CombinerConfig().cooling_off_minutes


def test_weighted_avg_method_available():
    cfg = CombinerConfig(combine_method="weighted_avg")
    c = ScoreCombiner(cfg)
    r = c.combine(anomaly_risk=15, xgb_fraud_proba=0.05, xgb_mule_proba=0.0,
                  mule_graph_risk=0.0, social_engineering_score=74.0)
    # weighted_avg dilutes the SE signal more than noisy_or -> higher trust.
    r_or = combiner.combine(anomaly_risk=15, xgb_fraud_proba=0.05, xgb_mule_proba=0.0,
                            mule_graph_risk=0.0, social_engineering_score=74.0)
    assert r.trust_score > r_or.trust_score


def test_templated_explanation_is_deterministic_and_offline():
    gen = TemplatedExplanationGenerator()
    features = {
        "trust_score": 30.0, "predicted_type": TYPE_SE,
        "recommended_action": ACTION_COOLING_OFF,
        "action_detail": {"delay_minutes": 20},
        "behavioral": {"anomaly_risk": 15, "xgb_fraud_proba": 0.05},
        "mule_graph": {"risk_flag": False, "account_count": 1},
        "social_engineering": {
            "score": 74, "is_likely": True,
            "contributions": {"first_transfer_to_payee": 22, "amount_vs_user_baseline_ratio": 28},
        },
    }
    breakdown = {"behavioral": 12, "mule_graph": 0, "social_engineering": 74}
    a = gen.explain(features, breakdown)
    b = gen.explain(features, breakdown)
    assert a == b  # deterministic
    assert "social-engineering" in a.lower()
    assert "cooling-off" in a.lower()
