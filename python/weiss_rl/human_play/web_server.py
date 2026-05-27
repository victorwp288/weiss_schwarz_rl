"""Small local JSON API for the React human-play client."""

from __future__ import annotations

import argparse
import importlib
import json
import mimetypes
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from weiss_rl.eval.god_search import GodSearchConfig
from weiss_rl.human_play.catalog import default_repo_root, list_candidate_runs, list_policies_for_run
from weiss_rl.human_play.decks import list_deck_presets
from weiss_rl.human_play.session import HumanPlayConfig, HumanPlaySession, HumanPlaySessionError


class SessionStore:
    def __init__(self) -> None:
        self._sessions: dict[str, HumanPlaySession] = {}

    def create(self, config: HumanPlayConfig) -> HumanPlaySession:
        session = HumanPlaySession(config)
        self._sessions[session.session_id] = session
        return session

    def get(self, session_id: str) -> HumanPlaySession:
        try:
            return self._sessions[session_id]
        except KeyError as exc:
            raise KeyError(f"unknown session id: {session_id}") from exc

    def close(self, session_id: str) -> None:
        session = self._sessions.pop(session_id, None)
        if session is not None:
            session.close()


def make_handler(*, store: SessionStore, static_dir: Path | None = None) -> type[BaseHTTPRequestHandler]:
    resolved_static_dir = None if static_dir is None else Path(static_dir).resolve()

    class HumanPlayRequestHandler(BaseHTTPRequestHandler):
        server_version = "WeissHumanPlay/0.1"

        def do_OPTIONS(self) -> None:  # noqa: N802
            self._send_json({}, status=HTTPStatus.NO_CONTENT)

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            try:
                if parsed.path == "/api/health":
                    self._send_json(_health_payload())
                    return
                if parsed.path == "/api/decks":
                    weiss_sim = importlib.import_module("weiss_sim")
                    self._send_json({"decks": [deck.to_json_dict() for deck in list_deck_presets(weiss_sim)]})
                    return
                if parsed.path == "/api/runs":
                    self._send_json(
                        {
                            "runs": [
                                run.to_json_dict()
                                for run in list_candidate_runs(repo_root=default_repo_root(), limit=120)
                            ]
                        }
                    )
                    return
                if parsed.path == "/api/policies":
                    query = parse_qs(parsed.query)
                    run_dir = _first_query_value(query, "run_dir")
                    if not run_dir:
                        raise ValueError("run_dir query parameter is required")
                    self._send_json(
                        {"policies": [policy.to_json_dict() for policy in list_policies_for_run(Path(run_dir))]}
                    )
                    return
                if parsed.path.startswith("/api/sessions/"):
                    session_id = parsed.path.removeprefix("/api/sessions/").strip("/")
                    self._send_json(store.get(session_id).current_state())
                    return
                if self._try_static(parsed.path):
                    return
                self._send_error(HTTPStatus.NOT_FOUND, "not found")
            except Exception as exc:  # pragma: no cover - exercised through integration tests/manual runs
                self._send_exception(exc)

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            try:
                payload = self._read_json()
                if parsed.path == "/api/sessions":
                    session = store.create(_config_from_payload(payload))
                    self._send_json(session.current_state(), status=HTTPStatus.CREATED)
                    return
                if parsed.path.startswith("/api/sessions/") and parsed.path.endswith("/actions"):
                    session_id = parsed.path.removeprefix("/api/sessions/").removesuffix("/actions").strip("/")
                    state = store.get(session_id).submit_human_action(
                        int(payload.get("action_id")),
                        client_view_hash64=None
                        if payload.get("client_view_hash64") is None
                        else str(payload.get("client_view_hash64")),
                    )
                    self._send_json(state)
                    return
                if parsed.path.startswith("/api/sessions/") and parsed.path.endswith("/close"):
                    session_id = parsed.path.removeprefix("/api/sessions/").removesuffix("/close").strip("/")
                    store.close(session_id)
                    self._send_json({"closed": True})
                    return
                self._send_error(HTTPStatus.NOT_FOUND, "not found")
            except Exception as exc:  # pragma: no cover - exercised through integration tests/manual runs
                self._send_exception(exc)

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            return

        def _read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("content-length", "0") or "0")
            if length <= 0:
                return {}
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("request body must be a JSON object")
            return payload

        def _try_static(self, path: str) -> bool:
            if resolved_static_dir is None:
                return False
            request_path = "/index.html" if path in {"", "/"} else path
            relative = request_path.lstrip("/")
            candidate = (resolved_static_dir / relative).resolve()
            if not str(candidate).startswith(str(resolved_static_dir)):
                self._send_error(HTTPStatus.FORBIDDEN, "forbidden")
                return True
            if not candidate.is_file():
                fallback = resolved_static_dir / "index.html"
                if not fallback.is_file():
                    return False
                candidate = fallback
            content_type = mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
            data = candidate.read_bytes()
            self.send_response(HTTPStatus.OK)
            self._send_common_headers(content_type=content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return True

        def _send_json(self, payload: dict[str, Any], *, status: HTTPStatus = HTTPStatus.OK) -> None:
            data = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
            self.send_response(status)
            self._send_common_headers(content_type="application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _send_error(self, status: HTTPStatus, message: str) -> None:
            self._send_json({"error": message}, status=status)

        def _send_exception(self, exc: Exception) -> None:
            if isinstance(exc, (HumanPlaySessionError, ValueError, KeyError)):
                self._send_error(HTTPStatus.BAD_REQUEST, str(exc))
                return
            self._send_error(HTTPStatus.INTERNAL_SERVER_ERROR, f"{type(exc).__name__}: {exc}")

        def _send_common_headers(self, *, content_type: str) -> None:
            self.send_header("Content-Type", content_type)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Cache-Control", "no-store")

    return HumanPlayRequestHandler


def run_server(*, host: str, port: int, static_dir: Path | None = None) -> None:
    server = ThreadingHTTPServer((host, int(port)), make_handler(store=SessionStore(), static_dir=static_dir))
    print(f"Serving human play API on http://{host}:{port}")
    try:
        server.serve_forever()
    finally:
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the Weiss Schwarz human-play web API")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--static-dir", type=Path, default=Path("web/human-play/dist"))
    args = parser.parse_args()
    static_dir = args.static_dir if args.static_dir.exists() else None
    run_server(host=str(args.host), port=int(args.port), static_dir=static_dir)


def _config_from_payload(payload: dict[str, Any]) -> HumanPlayConfig:
    run_dir = payload.get("run_dir")
    if not run_dir:
        raise ValueError("run_dir is required")
    return HumanPlayConfig(
        run_dir=Path(str(run_dir)),
        policy_id=str(payload.get("policy_id", "main_league_selected")),
        stack_config=_optional_path(payload.get("stack_config")),
        snapshot_registry_json=_optional_path(payload.get("snapshot_registry_json")),
        b1_baseline_run_dir=_optional_path(payload.get("b1_baseline_run_dir")),
        human_seat=int(payload.get("human_seat", 0)),
        seed=int(payload.get("seed", 20260521)),
        human_deck=str(payload.get("human_deck", "preset:main_deck_5hy_yotsuba_v1")),
        model_deck=str(payload.get("model_deck", "preset:main_deck_5hy_yotsuba_v1")),
        mode=str(payload.get("mode", "study")),
        model_sampling_algorithm=str(payload.get("model_sampling_algorithm", "model_argmax_pinned_v1")),
        artifact_root=_optional_path(payload.get("artifact_root")),
        top_k=int(payload.get("top_k", 5)),
        search_rollout_opponent_policy_id=str(payload.get("search_rollout_opponent_policy_id", "B0 RandomLegal")),
        god_search=GodSearchConfig.from_mapping(_optional_mapping(payload.get("god_search"))),
    )


def _optional_path(value: object) -> Path | None:
    text = "" if value is None else str(value).strip()
    return None if not text else Path(text)


def _optional_mapping(value: object) -> dict[str, Any] | None:
    return value if isinstance(value, dict) else None


def _first_query_value(query: dict[str, list[str]], name: str) -> str | None:
    values = query.get(name, [])
    if not values:
        return None
    text = str(values[0]).strip()
    return text or None


def _health_payload() -> dict[str, Any]:
    try:
        weiss_sim = importlib.import_module("weiss_sim")
    except ModuleNotFoundError:
        return {"ok": False, "weiss_sim": {"available": False, "human_decision_view": False}}
    return {
        "ok": callable(getattr(weiss_sim, "human_decision_view", None)),
        "weiss_sim": {
            "available": True,
            "version": getattr(weiss_sim, "__version__", None),
            "human_decision_view": callable(getattr(weiss_sim, "human_decision_view", None)),
            "file": getattr(weiss_sim, "__file__", None),
        },
    }


if __name__ == "__main__":
    main()
