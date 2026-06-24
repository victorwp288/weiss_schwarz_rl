# Human-play split deployment

The public human-play app should be split into three deployable surfaces:

1. Training/research repository: this repository owns training, evaluation,
   checkpoint selection, and thesis artifacts.
2. Human-play API: a Python service that loads `weiss_sim`, curated checkpoints,
   and live `HumanPlaySession` state.
3. Static web client: the Vite/React app under `web/human-play`, deployable to
   Vercel or any static host.

## Frontend

The frontend can be extracted into its own repository from `web/human-play`.
It is already configured as a Vite app. For a split deployment, configure:

```env
VITE_API_BASE=https://your-human-play-api.example.com
```

When `VITE_API_BASE` is empty, the UI keeps using same-origin `/api/...`, which
matches the current local Python-server flow.

`web/human-play/vercel.json` is intentionally small: Vercel can run `npm ci`,
`npm run build`, and publish `dist/`.

## Backend

The backend is not a static/serverless edge app. It is a stateful Python process
that imports the simulator, loads policy checkpoints, keeps live sessions in
memory, and writes optional transcripts. Deploy it as a small container service.

The starter Dockerfile is in `deploy/human-play-api/Dockerfile`.

Important environment variables:

| Variable | Purpose |
| --- | --- |
| `WEISS_HUMAN_PLAY_ALLOWED_ORIGINS` | Comma-separated browser origins allowed by CORS. Use the Vercel UI URL in production. Defaults to `*` for local development. |
| `WEISS_HUMAN_PLAY_REPO_ROOT` | Root containing the deployed `runs/` directory. Use this for mounted curated demo bundles. |
| `WEISS_CARD_DB` | Optional path to a scraped card DB JSONL file for card text/art lookup. |
| `WEISS_CARD_ART_CACHE` | Writable cache directory for proxied card scans. |

## Model bundle

Do not publish the full research `runs/` tree. Export a small demo bundle with
only selected public-safe runs and checkpoints:

```text
demo-bundle/
  runs/
    selected_demo_run/
      config_canonical.json
      manifest.json or spec_hash256.txt
      training/snapshots/registry.json
      training/snapshots/.../weights.pt
```

That bundle can be mounted into the backend container with
`WEISS_HUMAN_PLAY_REPO_ROOT=/data/weiss-demo`.

## Known limitation

There is not yet a maintained export command for public demo bundles. Until one
exists, build the bundle deliberately: copy only selected policy snapshots,
their config/spec metadata, and a pruned registry. The backend container should
not mount a full local research workspace in production.
