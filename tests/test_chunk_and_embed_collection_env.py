"""Contract tests for Qdrant collection recreate env logic (prod safety)."""

from __future__ import annotations

import pytest

from ingestion.chunk_and_embed import _should_recreate_qdrant_collection

_ENV_KEYS = (
    "RAILWAY_ENVIRONMENT",
    "ENVIRONMENT",
    "APP_ENV",
    "QDRANT_RECREATE_COLLECTION",
)


@pytest.mark.parametrize(
    ("env", "expected"),
    [
        pytest.param(
            {
                "RAILWAY_ENVIRONMENT": "production",
                "QDRANT_RECREATE_COLLECTION": "true",
            },
            False,
            id="railway_production_ignores_recreate_flag",
        ),
        pytest.param(
            {
                "ENVIRONMENT": "production",
                "QDRANT_RECREATE_COLLECTION": "true",
            },
            False,
            id="env_production_ignores_recreate_flag",
        ),
        pytest.param(
            {
                "ENVIRONMENT": "prod",
                "QDRANT_RECREATE_COLLECTION": "true",
            },
            False,
            id="env_prod_alias_ignores_recreate_flag",
        ),
        pytest.param(
            {"ENVIRONMENT": "development"},
            True,
            id="development_recreates",
        ),
        pytest.param(
            {"ENVIRONMENT": "dev"},
            True,
            id="dev_short_recreates",
        ),
        pytest.param({}, False, id="all_unset_defaults_no_recreate"),
        pytest.param(
            {"QDRANT_RECREATE_COLLECTION": "true"},
            True,
            id="explicit_true_non_prod",
        ),
        pytest.param(
            {
                "ENVIRONMENT": "development",
                "QDRANT_RECREATE_COLLECTION": "false",
            },
            False,
            id="explicit_false_overrides_development",
        ),
    ],
)
def test_should_recreate_qdrant_collection(
    monkeypatch: pytest.MonkeyPatch,
    env: dict[str, str],
    expected: bool,
) -> None:
    for key in _ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    assert _should_recreate_qdrant_collection() is expected
