# Forgefy self-hosted Zoom bot — one container per meeting.
#
# Two processes run inside: the C++ Meeting SDK client that joins the call and
# emits raw PCM, and a Python sidecar that streams that PCM to Deepgram and
# republishes transcripts onto the same Redis channel the Recall.ai path uses.

# ── base: runtime libraries the Meeting SDK links against ────────────────────
FROM ubuntu:24.04 AS base

SHELL ["/bin/bash", "-c"]
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libdbus-1-3 \
    libgbm1 \
    libgl1 \
    libglib2.0-0 \
    libx11-6 \
    libx11-xcb1 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-randr0 \
    libxcb-shape0 \
    libxcb-shm0 \
    libxcb-xfixes0 \
    libxcb-xtest0 \
    libxfixes3 \
    libssl3t64 \
    libgl1-mesa-dri \
    dbus \
    && rm -rf /var/lib/apt/lists/*

# The SDK refuses to initialize audio without a sound card. Containers have
# none, so PulseAudio provides a virtual sink at startup (see entrypoint.sh).
RUN apt-get update && apt-get install -y --no-install-recommends \
    alsa-utils \
    libasound2t64 \
    libasound2-plugins \
    pulseaudio \
    pulseaudio-utils \
    && rm -rf /var/lib/apt/lists/*

# ── builder: compile the SDK client ──────────────────────────────────────────
FROM base AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    pkgconf \
    libglib2.0-dev \
    libx11-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# lib/zoomsdk must already contain the unpacked Meeting SDK — it is licensed
# and cannot be fetched at build time. CMake fails with instructions if absent.
COPY lib/ ./lib/
COPY CMakeLists.txt ./
COPY src/ ./src/

RUN cmake -B build -S . -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build --parallel "$(nproc)"

# ── runtime ──────────────────────────────────────────────────────────────────
FROM base AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-venv \
    tini \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/forgefy

# PEP 668 marks Ubuntu's Python as externally managed; a venv keeps pip usable
# without disabling that protection.
ENV VIRTUAL_ENV=/opt/forgefy/venv
RUN python3 -m venv "$VIRTUAL_ENV"
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

COPY sidecar/requirements.txt ./sidecar/requirements.txt
RUN pip install --no-cache-dir -r sidecar/requirements.txt

COPY --from=builder /build/build/forgefy-zoom-bot ./bin/forgefy-zoom-bot
COPY --from=builder /build/lib/zoomsdk ./lib/zoomsdk
COPY sidecar/ ./sidecar/
COPY bin/entrypoint.sh ./bin/entrypoint.sh
RUN chmod +x ./bin/entrypoint.sh

# The SDK dynamically loads its own shared object by soname.
RUN ln -sf /opt/forgefy/lib/zoomsdk/libmeetingsdk.so \
           /opt/forgefy/lib/zoomsdk/libmeetingsdk.so.1
ENV LD_LIBRARY_PATH=/opt/forgefy/lib/zoomsdk:/opt/forgefy/lib/zoomsdk/qt_libs/Qt/lib

ENV FORGEFY_AUDIO_SOCKET=/tmp/forgefy/audio.sock \
    FORGEFY_BOT_BINARY=/opt/forgefy/bin/forgefy-zoom-bot \
    PYTHONUNBUFFERED=1

# tini reaps the SDK client when the sidecar exits, and forwards SIGTERM so
# `docker stop` makes the bot leave the meeting politely.
ENTRYPOINT ["/usr/bin/tini", "--", "/opt/forgefy/bin/entrypoint.sh"]
