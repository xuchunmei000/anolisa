"""Unit tests for Skill Ledger activation policy helpers."""

import pytest
from agent_sec_cli.skill_ledger.activation_policy import (
    ACTIVATION_POLICIES,
    ACTIVATION_POLICY_ALLOWED_SCAN_STATUSES,
    ACTIVATION_POLICY_LATEST_SCANNED,
    ACTIVATION_POLICY_PASS_ONLY,
    allowed_scan_statuses_for_policy,
    validate_activation_policy,
)


def test_activation_policies_are_derived_from_status_mapping():
    assert ACTIVATION_POLICIES == frozenset(ACTIVATION_POLICY_ALLOWED_SCAN_STATUSES)


@pytest.mark.parametrize(
    "policy",
    [ACTIVATION_POLICY_PASS_ONLY, ACTIVATION_POLICY_LATEST_SCANNED],
)
def test_validate_activation_policy_accepts_supported_policies(policy):
    assert validate_activation_policy(policy) == policy


@pytest.mark.parametrize("policy", ["unknown", ["pass_only"]])
def test_validate_activation_policy_rejects_invalid_policies(policy):
    with pytest.raises(ValueError, match="unsupported activation policy"):
        validate_activation_policy(policy)


def test_allowed_scan_statuses_for_policy():
    assert allowed_scan_statuses_for_policy(ACTIVATION_POLICY_PASS_ONLY) == frozenset(
        {"pass"}
    )
    assert allowed_scan_statuses_for_policy(
        ACTIVATION_POLICY_LATEST_SCANNED
    ) == frozenset({"pass", "warn", "deny"})
