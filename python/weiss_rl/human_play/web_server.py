"""Small local JSON API for the React human-play client."""

from __future__ import annotations

import argparse
import importlib
import json
import mimetypes
import os
import re
import time
import urllib.request
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote, unquote, urlparse

from weiss_rl.eval.search.god_search import GodSearchConfig
from weiss_rl.human_play.catalog import default_repo_root, list_candidate_runs, list_policies_for_run
from weiss_rl.human_play.decks import list_deck_presets
from weiss_rl.human_play.session import HumanPlayConfig, HumanPlaySession, HumanPlaySessionError

# ---------------------------------------------------------------- card art
# Card scans resolve through the simulator repo's scraped card DB
# (card_no -> image_url), falling back to scraping the public ws-tcg.com
# cardlist page. Images are downloaded once and cached on disk; the UI
# falls back to the procedural card face when a scan is unavailable (404).

_CARD_ART_IMG_RE = re.compile(r'src="(/wordpress/wp-content/images/card(?:images|list)/[^"]+)"')
_CARD_ART_SOURCES = ("https://en.ws-tcg.com", "https://ws-tcg.com")
_CARD_NO_RE = re.compile(r"^[A-Za-z0-9/_\- ]{3,40}$")
_ART_MISS_TTL_SECONDS = 24 * 3600.0

_card_image_url_index: dict[str, str] | None = None
_card_info_index: dict[str, dict[str, Any]] | None = None


def _card_db_candidates() -> list[Path]:
    candidates = []
    override = os.environ.get("WEISS_CARD_DB", "").strip()
    if override:
        candidates.append(Path(override))
    for root in (Path.cwd(), *Path.cwd().parents[:2]):
        candidates.append(root / "weiss-schwarz-simulator" / "scraper" / "out" / "cards.jsonl")
        candidates.append(root.parent / "weiss-schwarz-simulator" / "scraper" / "out" / "cards.jsonl")
    return candidates


def _load_card_db() -> tuple[dict[str, str], dict[str, dict[str, Any]]]:
    global _card_image_url_index, _card_info_index  # noqa: PLW0603 - lazy module cache
    if _card_image_url_index is not None and _card_info_index is not None:
        return _card_image_url_index, _card_info_index
    images: dict[str, str] = {}
    info: dict[str, dict[str, Any]] = {}
    db_path = next((candidate for candidate in _card_db_candidates() if candidate.is_file()), None)
    if db_path is not None:
        try:
            with db_path.open(encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    card_no = str(record.get("card_no") or "").strip()
                    if not card_no:
                        continue
                    image_url = str(record.get("image_url") or "").strip()
                    if image_url.startswith("http"):
                        images.setdefault(card_no, image_url)
                    info.setdefault(
                        card_no,
                        {
                            "card_no": card_no,
                            "name": record.get("name"),
                            "text": record.get("text"),
                            "rarity": record.get("rarity"),
                            "traits": record.get("traits"),
                            "expansion": record.get("expansion_raw"),
                        },
                    )
        except OSError:
            images, info = {}, {}
        print(f"card art: indexed {len(images)} image urls / {len(info)} card records from {db_path}")
    else:
        print("card art: no local card DB found; falling back to cardlist scraping")
    _card_image_url_index = images
    _card_info_index = info
    return images, info


def _load_card_image_index() -> dict[str, str]:
    return _load_card_db()[0]


def card_info_payload(card_no: str, card_id: int | None) -> dict[str, Any]:
    payload = dict(_load_card_db()[1].get(card_no.strip()) or {"card_no": card_no})
    if card_id is not None:
        try:
            weiss_sim = importlib.import_module("weiss_sim")
            card = weiss_sim.cards.get(int(card_id))
            payload["approx_ok"] = bool(getattr(card, "approx_ok", False))
            payload["strict_ok"] = bool(getattr(card, "strict_ok", False))
            payload.setdefault("name", str(getattr(card, "name", "") or "") or None)
        except Exception:  # noqa: BLE001 - sim lookup is best-effort decoration
            pass
    return payload


def _card_art_cache_dir() -> Path:
    override = os.environ.get("WEISS_CARD_ART_CACHE", "").strip()
    base = Path(override) if override else (Path.home() / ".cache" / "weiss_card_art")
    base.mkdir(parents=True, exist_ok=True)
    return base


def _card_art_safe_name(card_no: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", card_no.strip()).strip("_")


def _http_get(url: str, *, timeout: float = 12.0) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "weiss-rl-human-play/0.1 (local research tool)"})
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - fixed https hosts
        return response.read()


def fetch_card_art(card_no: str) -> Path | None:
    """Return a cached local path for the card scan, downloading it on first use."""
    if not _CARD_NO_RE.match(card_no):
        return None
    cache_dir = _card_art_cache_dir()
    safe = _card_art_safe_name(card_no)
    for suffix in (".png", ".jpg"):
        cached = cache_dir / f"{safe}{suffix}"
        if cached.is_file():
            return cached
    miss_marker = cache_dir / f"{safe}.miss"
    if miss_marker.is_file() and (time.time() - miss_marker.stat().st_mtime) < _ART_MISS_TTL_SECONDS:
        return None
    known_url = _load_card_image_index().get(card_no.strip())
    if known_url:
        try:
            data = _http_get(known_url)
            if data:
                suffix = ".jpg" if known_url.lower().endswith((".jpg", ".jpeg")) else ".png"
                target = cache_dir / f"{safe}{suffix}"
                target.write_bytes(data)
                miss_marker.unlink(missing_ok=True)
                return target
        except Exception:  # noqa: BLE001 - fall through to page scraping
            pass
    for host in _CARD_ART_SOURCES:
        try:
            page = _http_get(f"{host}/cardlist/?cardno={quote(card_no, safe='')}").decode("utf-8", "replace")
            match = _CARD_ART_IMG_RE.search(page)
            if not match:
                continue
            image_path = match.group(1)
            data = _http_get(f"{host}{image_path}")
            if not data:
                continue
            suffix = ".jpg" if image_path.lower().endswith((".jpg", ".jpeg")) else ".png"
            target = cache_dir / f"{safe}{suffix}"
            target.write_bytes(data)
            miss_marker.unlink(missing_ok=True)
            return target
        except Exception:  # noqa: BLE001 - network best-effort; UI has a fallback face
            continue
    miss_marker.write_bytes(b"")
    return None


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


def make_handler(
    *,
    store: SessionStore,
    static_dir: Path | None = None,
    allowed_origins: tuple[str, ...] | None = None,
) -> type[BaseHTTPRequestHandler]:
    resolved_static_dir = None if static_dir is None else Path(static_dir).resolve()
    cors_origins = allowed_origins or _allowed_origins_from_env()

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
                if parsed.path.startswith("/api/card-info/"):
                    card_no = unquote(parsed.path.removeprefix("/api/card-info/").strip("/"))
                    query = parse_qs(parsed.query)
                    raw_card_id = _first_query_value(query, "card_id")
                    card_id = int(raw_card_id) if raw_card_id and raw_card_id.lstrip("-").isdigit() else None
                    self._send_json(card_info_payload(card_no, card_id))
                    return
                if parsed.path.startswith("/api/card-art/"):
                    card_no = unquote(parsed.path.removeprefix("/api/card-art/").strip("/"))
                    art_path = fetch_card_art(card_no)
                    if art_path is None:
                        self._send_error(HTTPStatus.NOT_FOUND, f"no card art for {card_no}")
                        return
                    self._send_file(art_path, cache_control="public, max-age=604800, immutable")
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
                if parsed.path.startswith("/api/sessions/") and parsed.path.endswith("/step"):
                    session_id = parsed.path.removeprefix("/api/sessions/").removesuffix("/step").strip("/")
                    self._send_json(store.get(session_id).step_model_decision())
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
            self._send_file(candidate)
            return True

        def _send_file(self, path: Path, *, cache_control: str | None = None) -> None:
            content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            data = path.read_bytes()
            self.send_response(HTTPStatus.OK)
            self._send_common_headers(content_type=content_type, cache_control=cache_control)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

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

        def _send_common_headers(self, *, content_type: str, cache_control: str | None = None) -> None:
            self.send_header("Content-Type", content_type)
            allowed_origin = _cors_origin_for_request(self.headers.get("Origin"), cors_origins)
            if allowed_origin:
                self.send_header("Access-Control-Allow-Origin", allowed_origin)
                self.send_header("Vary", "Origin")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Cache-Control", cache_control or "no-store")

    return HumanPlayRequestHandler


def run_server(
    *,
    host: str,
    port: int,
    static_dir: Path | None = None,
    allowed_origins: tuple[str, ...] | None = None,
) -> None:
    server = ThreadingHTTPServer(
        (host, int(port)),
        make_handler(store=SessionStore(), static_dir=static_dir, allowed_origins=allowed_origins),
    )
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
    parser.add_argument(
        "--allowed-origin",
        action="append",
        default=None,
        help="Browser origin allowed to call the API; repeatable. Defaults to WEISS_HUMAN_PLAY_ALLOWED_ORIGINS or '*'.",
    )
    args = parser.parse_args()
    static_dir = args.static_dir if args.static_dir.exists() else None
    allowed_origins = tuple(args.allowed_origin) if args.allowed_origin else None
    run_server(host=str(args.host), port=int(args.port), static_dir=static_dir, allowed_origins=allowed_origins)


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
        spectate=bool(payload.get("spectate", False)),
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


def _allowed_origins_from_env() -> tuple[str, ...]:
    raw = os.environ.get("WEISS_HUMAN_PLAY_ALLOWED_ORIGINS", "").strip()
    if not raw:
        return ("*",)
    origins = tuple(origin.strip().rstrip("/") for origin in raw.split(",") if origin.strip())
    return origins or ("*",)


def _cors_origin_for_request(origin: str | None, allowed_origins: tuple[str, ...]) -> str | None:
    normalized = "" if origin is None else str(origin).strip().rstrip("/")
    if "*" in allowed_origins:
        return normalized or "*"
    return normalized if normalized in allowed_origins else None


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
