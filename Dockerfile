FROM python:3.11-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg && \
    rm -rf /var/lib/apt/lists/*

RUN useradd --create-home claudetube
USER claudetube
WORKDIR /home/claudetube/app

COPY --chown=claudetube:claudetube . .

RUN pip install --no-cache-dir ".[mcp]"

# Cache directory for downloaded videos
RUN mkdir -p /home/claudetube/.claude/video_cache
VOLUME ["/home/claudetube/.claude/video_cache"]

ENTRYPOINT ["claudetube-mcp"]
