# syntax=docker/dockerfile:1
# Forgefy API image - lean, SDK-free, multi-stage.
#
# The API process (uvicorn) never invokes Node, OpenJDK, the Android SDK or
# Flutter; only the build worker (Dockerfile.worker) needs those, so they are
# intentionally NOT installed here. Compiled Python wheels are built in the
# `wheel-builder` stage so the gcc/cpp/binutils/perl toolchain never ships in
# the final image.
#
# Final image is roughly ~1 GB vs ~8.2 GB before (estimated).

# ---- Stage 1: build Python wheels ----------------------------------------
FROM python:3.14-slim AS wheel-builder

WORKDIR /build

# Compiler toolchain - only needed to build native wheels (greenlet, orjson,
# cryptography, cffi, bcrypt, httptools, watchfiles, pydantic-core, ...).
# These are NOT copied into the runtime stage. apt lists are removed in the
# same layer they are created in, so they are never captured.
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      gcc \
      cpp \
      binutils \
      perl \
      make \
      xz-utils \
      ca-certificates \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
# pip cache cleaned in the SAME layer so it never lands in the image.
RUN pip install --no-cache-dir --upgrade pip \
  && pip install --no-cache-dir -r requirements.txt \
  && rm -rf /root/.cache/pip

# ---- Stage 2: slim runtime ------------------------------------------------
FROM python:3.14-slim AS runtime

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Minimal runtime system packages: curl is used by the HEALTHCHECK, and
# ca-certificates for outbound TLS. No other packages are needed at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
      curl \
      ca-certificates \
  && rm -rf /var/lib/apt/lists/*

# Python site-packages + console scripts built in stage 1. Same base image, so
# compiled extensions match the runtime glibc exactly.
COPY --from=wheel-builder /usr/local/lib/python3.14/site-packages /usr/local/lib/python3.14/site-packages
COPY --from=wheel-builder /usr/local/bin /usr/local/bin

# ---- Playwright browser (optional, OFF by default) -------------------------
# Meet/Teams go through Recall.ai; the legacy Playwright MeetConnector is not
# wired in. The `playwright` pip package stays importable, but the Chromium
# browser is only downloaded when INSTALL_PLAYWRIGHT_BROWSERS=true.
ARG INSTALL_PLAYWRIGHT_BROWSERS=false
RUN if [ "$INSTALL_PLAYWRIGHT_BROWSERS" = "true" ]; then \
      playwright install chromium --with-deps ; \
    fi

# ---- Application -----------------------------------------------------------
COPY . .

# Non-root user. /app stays root-owned (read-only at runtime, as before).
RUN useradd -m -s /bin/bash forgefy

USER forgefy
ENV HOME=/home/forgefy

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/health || exit 1

# Start Forgefy API
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} \
    --workers ${WEB_CONCURRENCY:-4} \
    --proxy-headers --forwarded-allow-ips="*"