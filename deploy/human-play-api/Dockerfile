FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    WEISS_CARD_ART_CACHE=/data/card_art \
    WEISS_HUMAN_PLAY_REPO_ROOT=/data/weiss-demo

WORKDIR /app

COPY pyproject.toml README.md ./
COPY python ./python

RUN python -m pip install --no-cache-dir --upgrade pip && \
    python -m pip install --no-cache-dir ".[sim]"

EXPOSE 8765

CMD ["python", "-m", "weiss_rl.human_play.web_server", "--host", "0.0.0.0", "--port", "8765", "--static-dir", "/app/static"]
