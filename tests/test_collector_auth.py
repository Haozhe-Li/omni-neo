"""The collector key: off by default, exact match, never a user identity.

    venv/bin/python3.12 -m pytest tests/test_collector_auth.py -q
"""

import pytest
from fastapi import HTTPException

from core.auth import COLLECTOR_USER_PREFIX, get_collector, resolve_user


def call(key, annotator, monkeypatch, env="s3cret"):
    if env is None:
        monkeypatch.delenv("COLLECTOR_API_KEY", raising=False)
    else:
        monkeypatch.setenv("COLLECTOR_API_KEY", env)
    return get_collector(x_collector_key=key, x_collector_id=annotator)


def test_disabled_without_a_configured_key(monkeypatch):
    with pytest.raises(HTTPException) as e:
        call("anything", "alice", monkeypatch, env=None)
    assert e.value.status_code == 503


@pytest.mark.parametrize("key", [None, "", "wrong", "s3cret ", "S3CRET", "s3cre"])
def test_wrong_key(key, monkeypatch):
    with pytest.raises(HTTPException) as e:
        call(key, "alice", monkeypatch)
    assert e.value.status_code == 401


def test_empty_configured_key_never_matches_an_empty_header(monkeypatch):
    with pytest.raises(HTTPException) as e:
        call("", "alice", monkeypatch, env="")
    assert e.value.status_code == 503


def test_returns_a_partitioned_owner_id(monkeypatch):
    assert call("s3cret", "Alice", monkeypatch) == f"{COLLECTOR_USER_PREFIX}alice"


@pytest.mark.parametrize("annotator", [None, "", "a b", "../x", "x" * 33, "ü"])
def test_annotator_id_shape(annotator, monkeypatch):
    with pytest.raises(HTTPException) as e:
        call("s3cret", annotator, monkeypatch)
    assert e.value.status_code == 400


def test_the_key_is_not_a_chat_identity(monkeypatch):
    # /chat's identity resolution knows nothing about the collector key or prefix
    monkeypatch.setenv("COLLECTOR_API_KEY", "s3cret")
    for guest in (None, "collector_alice", "s3cret"):
        with pytest.raises(HTTPException):
            resolve_user(bearer_token=None, guest_id=guest)
