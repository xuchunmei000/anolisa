"""Activation policy definitions shared by config and resolver."""

from typing import Any

ACTIVATION_POLICY_PASS_ONLY = "pass_only"
ACTIVATION_POLICY_PASS_WARN_ONLY = "pass_warn_only"
ACTIVATION_POLICY_LATEST_SCANNED = "latest_scanned"
DEFAULT_ACTIVATION_POLICY = ACTIVATION_POLICY_LATEST_SCANNED

ACTIVATION_POLICY_ALLOWED_SCAN_STATUSES: dict[str, frozenset[str]] = {
    ACTIVATION_POLICY_PASS_ONLY: frozenset({"pass"}),
    ACTIVATION_POLICY_PASS_WARN_ONLY: frozenset({"pass", "warn"}),
    ACTIVATION_POLICY_LATEST_SCANNED: frozenset({"pass", "warn", "deny"}),
}
ACTIVATION_POLICIES = frozenset(ACTIVATION_POLICY_ALLOWED_SCAN_STATUSES)


def validate_activation_policy(policy: Any) -> str:
    """Return a valid activation policy or raise ``ValueError``."""
    if not isinstance(policy, str) or policy not in ACTIVATION_POLICIES:
        allowed = ", ".join(sorted(ACTIVATION_POLICIES))
        raise ValueError(
            f"unsupported activation policy: {policy}; expected one of: {allowed}"
        )
    return policy


def allowed_scan_statuses_for_policy(policy: Any) -> frozenset[str]:
    """Return scan statuses that the activation policy may expose."""
    return ACTIVATION_POLICY_ALLOWED_SCAN_STATUSES[validate_activation_policy(policy)]
