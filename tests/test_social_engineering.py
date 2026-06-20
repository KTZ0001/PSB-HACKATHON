"""Phase 2 tests for the social-engineering risk-policy engine + profile store."""
from src.scoring.social_engineering import (
    RuleWeightedSocialEngineeringEngine,
    SocialEngineeringConfig,
    signal_vector,
    SE_SIGNAL_NAMES,
)
from src.features.user_profile import UserProfileStore

engine = RuleWeightedSocialEngineeringEngine()

NORMAL = {
    "first_transfer_to_payee": False,
    "payee_age_hours": 5000.0,
    "amount_vs_user_baseline_ratio": 1.0,
    "txn_velocity_short_window": 1,
    "failed_attempts_before_success": 0,
    "is_new_high_risk_country_or_geo": False,
    "night_time_transaction": False,
}

# The thesis case: normal device/geo, suspicious money shape.
COACHED_VICTIM = {
    "first_transfer_to_payee": True,
    "payee_age_hours": 0.2,
    "amount_vs_user_baseline_ratio": 5.2,
    "txn_velocity_short_window": 1,
    "failed_attempts_before_success": 2,
    "is_new_high_risk_country_or_geo": False,
    "night_time_transaction": False,
}


def test_normal_transaction_scores_low():
    res = engine.score(NORMAL)
    assert res.social_engineering_score < 10
    assert res.is_likely_social_engineering is False


def test_coached_victim_flagged_despite_clean_device_geo():
    res = engine.score(COACHED_VICTIM)
    # Flagged even though geo + night signals are OFF.
    assert res.is_likely_social_engineering is True
    assert res.social_engineering_score >= 50
    assert res.contributions["is_new_high_risk_country_or_geo"] == 0.0
    assert res.contributions["night_time_transaction"] == 0.0
    # The flag is driven by the money-movement signals.
    assert res.contributions["first_transfer_to_payee"] > 0
    assert res.contributions["amount_vs_user_baseline_ratio"] > 0


def test_score_is_bounded_and_monotonic_in_amount():
    low = dict(NORMAL, amount_vs_user_baseline_ratio=1.0)
    high = dict(NORMAL, amount_vs_user_baseline_ratio=10.0)
    r_low, r_high = engine.score(low), engine.score(high)
    assert 0 <= r_low.social_engineering_score <= 100
    assert 0 <= r_high.social_engineering_score <= 100
    assert r_high.social_engineering_score > r_low.social_engineering_score


def test_config_is_tunable():
    # A 2x-baseline amount contributes ~9.3 pts; a strict threshold flags it,
    # the default threshold (50) does not -> demonstrates the config knob works.
    strict = SocialEngineeringConfig(flag_threshold=8.0)
    engine_strict = RuleWeightedSocialEngineeringEngine(strict)
    mild = dict(NORMAL, amount_vs_user_baseline_ratio=2.0)
    assert engine_strict.score(mild).is_likely_social_engineering is True
    assert engine.score(mild).is_likely_social_engineering is False  # default thr 50


def test_signal_vector_shape_and_order():
    vec = signal_vector(COACHED_VICTIM)
    assert vec.shape == (len(SE_SIGNAL_NAMES),)
    assert vec[0] == 1.0  # first_transfer_to_payee True


def test_profile_store_detects_first_transfer_above_baseline():
    store = UserProfileStore()
    store.seed_baseline("u1", [100, 100, 100])
    signals = store.compute_signals(
        user_id="u1", amount=500.0, payee_id="new_payee",
        timestamp=10.0, country="UK", hour=14, failed_attempts_before_success=1,
    )
    assert signals["first_transfer_to_payee"] is True
    assert signals["payee_age_hours"] == 0.0
    assert signals["amount_vs_user_baseline_ratio"] == 5.0
    assert signals["is_new_high_risk_country_or_geo"] is False  # UK not high-risk
    # After update, the same payee is no longer 'first'.
    store.update("u1", 500.0, "new_payee", 10.0)
    again = store.compute_signals(
        user_id="u1", amount=120.0, payee_id="new_payee", timestamp=4000.0,
        country="UK", hour=14,
    )
    assert again["first_transfer_to_payee"] is False
    assert again["payee_age_hours"] > 0


def test_profile_store_flags_high_risk_country():
    store = UserProfileStore()
    store.seed_baseline("u2", [100])
    signals = store.compute_signals(
        user_id="u2", amount=100.0, payee_id="p", timestamp=0.0, country="NG", hour=14,
    )
    assert signals["is_new_high_risk_country_or_geo"] is True
