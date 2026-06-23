# Shōbu 勝負 — Weiss Schwarz human-play web client

A React UI for playing Weiss Schwarz against a trained policy via the local
`weiss_rl.human_play` API.

The UI can also run as a static deployment pointed at a hosted API with
`VITE_API_BASE`. That is the intended shape for a public Vercel frontend plus a
separate Python backend container.

## Important: the API needs the project virtualenv

The status pill shows **"Simulator API unavailable"** when the browser can't reach
a healthy API. The usual cause is starting the API with a Python that can't import
`weiss_sim`. **Always launch the server with the project venv.**

## Run it (one command — recommended)

Build the UI once, then let the Python server serve it _and_ the API on one port
(same origin, no proxy):

```bash
# from the repo root
npm --prefix web/human-play install        # first time only
npm --prefix web/human-play run build
.venv/Scripts/python -m weiss_rl.human_play.web_server --static-dir web/human-play/dist
#   → open http://127.0.0.1:8765
```

On macOS/Linux use `.venv/bin/python` instead of `.venv/Scripts/python`.

## Run it (dev mode with hot reload)

Two processes. The Vite dev server proxies `/api` → `http://127.0.0.1:8765`
(see [`vite.config.ts`](./vite.config.ts)).

```bash
# terminal 1 — the API (must use the venv)
.venv/Scripts/python -m weiss_rl.human_play.web_server --port 8765

# terminal 2 — the UI with hot reload
npm --prefix web/human-play run dev
#   → open the URL Vite prints (http://127.0.0.1:5174 by default)
```

If your API runs on a different host/port, set `VITE_API_PROXY`, e.g.
`VITE_API_PROXY=http://127.0.0.1:9000 npm run dev`.

## Split deployment

For Vercel or another static host, set:

```bash
VITE_API_BASE=https://your-human-play-api.example.com
```

Then build/deploy this directory as a normal Vite app:

```bash
npm ci
npm run build
```

The Python API must allow the static site's browser origin, for example:

```bash
WEISS_HUMAN_PLAY_ALLOWED_ORIGINS=https://your-ui.vercel.app \
python -m weiss_rl.human_play.web_server --host 0.0.0.0 --port 8765
```

See `../../docs/human_play_deployment.md` for the split deployment shape and
`../../deploy/human-play-api/README.md` for the container/backend side.

## Test & type-check

```bash
npm --prefix web/human-play run test      # vitest
npm --prefix web/human-play run build     # tsc -b + vite build
```
