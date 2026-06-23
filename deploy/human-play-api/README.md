# Human-play API container

This image runs the Python `weiss_rl.human_play.web_server` service. It is the
stateful simulator/model API for the static web client.

Build from the repository root:

```bash
docker build -f deploy/human-play-api/Dockerfile -t weiss-play-api .
```

Run locally with a curated artifact root mounted at `/data/weiss-demo`:

```bash
docker run --rm -p 8765:8765 \
  -e WEISS_HUMAN_PLAY_ALLOWED_ORIGINS=http://127.0.0.1:5174,https://your-ui.vercel.app \
  -e WEISS_HUMAN_PLAY_REPO_ROOT=/data/weiss-demo \
  -v /absolute/path/to/demo-bundle:/data/weiss-demo:ro \
  weiss-play-api
```

The mounted demo bundle should look like the repository surfaces the API already
expects:

```text
demo-bundle/
  runs/
    selected_demo_run/
      config_canonical.json
      manifest.json or spec_hash256.txt
      training/snapshots/registry.json
      training/snapshots/.../weights.pt
```

The image intentionally does not copy local `runs/` by default. Export a small,
public-safe model bundle from the training repo and mount it into the container
instead of publishing every experiment artifact.
