from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest
from weiss_rl.human_play.catalog import default_repo_root, list_candidate_runs, list_policies_for_run
from weiss_rl.human_play.decks import list_deck_presets, preset_id, preset_name
from weiss_rl.human_play.session import (
    HumanPlaySessionError,
    _enrich_card_labels,
    _normalize_policy_id,
    _normalize_view,
)
from weiss_rl.human_play.transcript import DecisionRecord, HumanPlayTranscript
from weiss_rl.human_play.web_server import SessionStore, _config_from_payload, make_handler


class _FakeCards:
    def presets(self) -> list[str]:
        return ["main_deck_5hy_yotsuba_v1", "tiny_custom_v1"]

    def preset_metadata(self) -> dict[str, dict[str, str]]:
        return {"tiny_custom_v1": {"source": "unit", "min_rules_profile": "approx"}}

    def describe_deck(self, name: str, *, rules_profile: str, card_pool: str) -> dict[str, Any]:
        assert rules_profile == "approx"
        assert card_pool == "all"
        if name == "main_deck_5hy_yotsuba_v1":
            return {
                "counts": [
                    {"id": "a", "name": "Yotsuba A", "count": 4},
                    {"id": "b", "name": "Yotsuba B", "count": 2},
                ]
            }
        return {"cards": [{"id": "x", "name": "Custom X"}, {"id": "y", "name": "Custom Y"}]}


class _FakeWeissSim:
    cards = _FakeCards()


def test_deck_presets_have_human_names_and_counts() -> None:
    decks = {deck.deck_id: deck for deck in list_deck_presets(_FakeWeissSim())}

    main = decks["preset:main_deck_5hy_yotsuba_v1"]
    assert main.label == "Yotsuba thesis deck"
    assert main.role == "primary thesis deck"
    assert main.card_count == 6
    assert main.unique_card_count == 2
    assert main.sample_cards == ("Yotsuba A", "Yotsuba B")

    custom = decks["preset:tiny_custom_v1"]
    assert custom.label == "Tiny Custom"
    assert custom.role == "freeplay deck"
    assert custom.source == "unit"
    assert custom.card_count == 2


def test_preset_helpers_are_round_trip_safe() -> None:
    assert preset_id("foo") == "preset:foo"
    assert preset_id("preset:foo") == "preset:foo"
    assert preset_name("preset:foo") == "foo"
    assert preset_name("foo") == "foo"


def test_normalize_view_requires_legal_order_match() -> None:
    view = {
        "legal_action_ids": [2, 1],
        "legal_actions": [{"action_id": 1, "label": "First"}, {"action_id": 2, "label": "Second"}],
    }

    with pytest.raises(HumanPlaySessionError, match="legal_action_ids"):
        _normalize_view(view)


def test_enrich_card_labels_adds_catalog_names_without_changing_shape() -> None:
    class Cards:
        def get(self, card_id: int) -> Any:
            return type("Card", (), {"name": f"Known {card_id}", "card_no": f"CARD-{card_id}"})()

    payload = {"zones": {"hand": {"cards": [{"card": {"card_id": 17, "level": 1}}]}}}
    enriched = _enrich_card_labels(payload, type("Sim", (), {"cards": Cards()})())

    card = enriched["zones"]["hand"]["cards"][0]["card"]
    assert card["name"] == "Known 17"
    assert card["card_no"] == "CARD-17"
    assert card["level"] == 1


def test_transcript_writes_manifest_decisions_events_and_postgame(tmp_path: Path) -> None:
    transcript = HumanPlayTranscript(tmp_path / "session", manifest={"session_id": "abc"})
    transcript.append_event({"event": "started"})
    transcript.append_decision(
        DecisionRecord(
            decision_index=0,
            actor_seat=0,
            actor_kind="human",
            action_id=7,
            action_label="Play card",
            legal_action_ids=(7, 9),
            decision_id=3,
            decision_kind="main",
            view_hash64="view",
            legal_fingerprint64="legal",
            elapsed_ms=None,
        )
    )
    transcript.write_postgame_report({"status": "complete", "terminal": True, "decision_count": 1, "winner_seat": 0})

    manifest = json.loads(transcript.manifest_path.read_text(encoding="utf-8"))
    decision = json.loads(transcript.decisions_path.read_text(encoding="utf-8").strip())
    event = json.loads(transcript.events_path.read_text(encoding="utf-8").strip())

    assert manifest["schema_version"] == "human_play_manifest_v1"
    assert decision["legal_action_ids"] == [7, 9]
    assert decision["action_label"] == "Play card"
    assert event["event"] == "started"
    assert "winner_seat" in transcript.postgame_path.read_text(encoding="utf-8")


def test_config_from_payload_maps_user_choices() -> None:
    config = _config_from_payload(
        {
            "run_dir": "runs/example",
            "policy_id": "policy_000003",
            "human_seat": 1,
            "human_deck": "preset:aggro_deck_5hy_nino_v1",
            "model_deck": "preset:control_deck_jj_s66_v1",
            "artifact_root": "runs/example/human_play",
            "top_k": 7,
            "search_rollout_opponent_policy_id": "B2 HeuristicPublic",
            "god_search": {"mode": "same_world_prefix_rollout", "top_k": 3},
        }
    )

    assert config.run_dir == Path("runs/example")
    assert config.policy_id == "policy_000003"
    assert config.human_seat == 1
    assert config.human_deck == "preset:aggro_deck_5hy_nino_v1"
    assert config.model_deck == "preset:control_deck_jj_s66_v1"
    assert config.artifact_root == Path("runs/example/human_play")
    assert config.top_k == 7
    assert config.search_rollout_opponent_policy_id == "B2 HeuristicPublic"
    assert config.god_search.enabled is True
    assert config.god_search.top_k == 3


def test_config_from_payload_requires_run_dir() -> None:
    with pytest.raises(ValueError, match="run_dir is required"):
        _config_from_payload({})


def test_auto_policy_id_resolves_to_latest_snapshot(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "example"
    _write_registry(run_dir)

    assert _normalize_policy_id(run_dir, "main_league_selected") == "policy_000002"
    assert _normalize_policy_id(run_dir, "latest") == "policy_000002"
    assert _normalize_policy_id(run_dir, "policy_manual") == "policy_manual"


def test_auto_policy_id_prefers_explicit_main_selected_snapshot(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "example"
    selected_policy_id = "seed_demo_main_league_selected"
    _write_registry(
        run_dir,
        snapshots=[
            _snapshot_row("policy_000010", update=10, sha="a"),
            _snapshot_row(selected_policy_id, update=0, sha="b"),
            _snapshot_row("b1_noleague_baseline", update=50, sha="c"),
        ],
    )

    assert _normalize_policy_id(run_dir, "main_league_selected") == selected_policy_id
    assert _normalize_policy_id(run_dir, "auto") == selected_policy_id
    assert _normalize_policy_id(run_dir, "latest") == "b1_noleague_baseline"


def test_run_catalog_lists_runs_and_policies(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "main_league_test_run"
    run_dir.mkdir(parents=True)
    (run_dir / "config_canonical.json").write_text("{}\n", encoding="utf-8")
    _write_registry(run_dir)

    runs = list_candidate_runs(repo_root=tmp_path)
    assert len(runs) == 1
    assert runs[0].label == "main league test run"
    assert runs[0].policy_count == 2
    assert runs[0].default_policy_id == "main_league_selected"

    policies = list_policies_for_run(run_dir)
    assert policies[0].policy_id == "main_league_selected"
    assert policies[0].selected_by_default is True
    assert any(policy.policy_id == "B0 RandomLegal" for policy in policies)
    assert any(policy.policy_id == "policy_000002" and policy.label == "Policy 2 (update 20)" for policy in policies)


def test_run_catalog_root_can_be_configured_for_deployed_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("WEISS_HUMAN_PLAY_REPO_ROOT", str(tmp_path))

    assert default_repo_root() == tmp_path.resolve()


class _FakeSession:
    session_id = "session-1"

    def __init__(self) -> None:
        self.action_payload: dict[str, Any] | None = None
        self.closed = False

    def current_state(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "human_turn": True,
            "terminal": False,
            "view": {
                "view_hash64": "hash",
                "legal_action_ids": [11],
                "legal_actions": [{"action_id": 11, "label": "Legal move"}],
            },
        }

    def submit_human_action(self, action_id: int, *, client_view_hash64: str | None = None) -> dict[str, Any]:
        self.action_payload = {"action_id": action_id, "client_view_hash64": client_view_hash64}
        return self.current_state()

    def close(self) -> None:
        self.closed = True


class _FakeStore(SessionStore):
    def __init__(self) -> None:
        self.session = _FakeSession()

    def create(self, config: Any) -> _FakeSession:
        assert config.run_dir == Path("runs/example")
        return self.session

    def get(self, session_id: str) -> _FakeSession:
        assert session_id == "session-1"
        return self.session

    def close(self, session_id: str) -> None:
        assert session_id == "session-1"
        self.session.close()


def test_web_handler_creates_session_and_submits_legal_action() -> None:
    store = _FakeStore()
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(store=store, static_dir=None))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_port}"
    try:
        created = _post_json(f"{base_url}/api/sessions", {"run_dir": "runs/example"})
        assert created["session_id"] == "session-1"
        assert created["view"]["legal_action_ids"] == [11]

        after_action = _post_json(
            f"{base_url}/api/sessions/session-1/actions",
            {"action_id": 11, "client_view_hash64": "hash"},
        )
        assert after_action["human_turn"] is True
        assert store.session.action_payload == {"action_id": 11, "client_view_hash64": "hash"}

        closed = _post_json(f"{base_url}/api/sessions/session-1/close", {})
        assert closed == {"closed": True}
        assert store.session.closed is True
    finally:
        server.shutdown()
        server.server_close()


def test_web_handler_returns_json_errors() -> None:
    store = _FakeStore()
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(store=store, static_dir=None))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            _post_json(f"http://127.0.0.1:{server.server_port}/api/sessions", {})
        assert exc_info.value.code == 400
        payload = json.loads(exc_info.value.read().decode("utf-8"))
        assert payload["error"] == "run_dir is required"
    finally:
        server.shutdown()
        server.server_close()


def test_web_handler_restricts_cors_to_configured_origins() -> None:
    store = _FakeStore()
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0),
        make_handler(
            store=store,
            static_dir=None,
            allowed_origins=("https://weiss-play.vercel.app", "http://127.0.0.1:5174"),
        ),
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_port}"
    try:
        allowed = _options(f"{base_url}/api/sessions", origin="https://weiss-play.vercel.app")
        assert allowed.headers["Access-Control-Allow-Origin"] == "https://weiss-play.vercel.app"
        assert allowed.headers["Vary"] == "Origin"

        denied = _options(f"{base_url}/api/sessions", origin="https://example.invalid")
        assert "Access-Control-Allow-Origin" not in denied.headers
    finally:
        server.shutdown()
        server.server_close()


def _options(url: str, *, origin: str) -> urllib.response.addinfourl:
    request = urllib.request.Request(url, headers={"Origin": origin}, method="OPTIONS")
    return urllib.request.urlopen(request, timeout=5)


def _post_json(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        loaded = json.loads(response.read().decode("utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def _snapshot_row(policy_id: str, *, update: int, sha: str) -> dict[str, Any]:
    return {
        "policy_id": policy_id,
        "update": update,
        "weights_sha256": sha,
        "path": f"training/snapshots/{policy_id}/weights.pt",
        "created_utc": "2026-05-21T00:00:00+00:00",
    }


def _write_registry(run_dir: Path, *, snapshots: list[dict[str, Any]] | None = None) -> None:
    registry_path = run_dir / "training" / "snapshots" / "registry.json"
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "recent_size": 24,
                "champion_size": 4,
                "champion_snapshots": [],
                "pinned_snapshots": [],
                "snapshots": snapshots
                or [
                    _snapshot_row("policy_000001", update=10, sha="a"),
                    _snapshot_row("policy_000002", update=20, sha="b"),
                ],
            }
        ),
        encoding="utf-8",
    )
