"""Release metadata validation tests."""

from __future__ import annotations

import json

import pytest

from dev.check_release import main


def write_release_event(tmp_path, *, prerelease: bool):
    event_path = tmp_path / "event.json"
    event_path.write_text(
        json.dumps({"release": {"prerelease": prerelease}}),
        encoding="utf-8",
    )
    return event_path


def test_alpha_release_metadata_is_valid(monkeypatch, tmp_path) -> None:
    event_path = write_release_event(tmp_path, prerelease=True)
    monkeypatch.setenv("GITHUB_REF_NAME", "v0.0.1a0")
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(event_path))

    assert main() == 0


def test_release_tag_must_match_package_version(monkeypatch, tmp_path) -> None:
    event_path = write_release_event(tmp_path, prerelease=True)
    monkeypatch.setenv("GITHUB_REF_NAME", "v0.0.2a0")
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(event_path))

    with pytest.raises(RuntimeError, match="must be 'v0.0.1a0'"):
        main()


def test_alpha_version_requires_github_prerelease(monkeypatch, tmp_path) -> None:
    event_path = write_release_event(tmp_path, prerelease=False)
    monkeypatch.setenv("GITHUB_REF_NAME", "v0.0.1a0")
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(event_path))

    with pytest.raises(RuntimeError, match="pre-release state"):
        main()
