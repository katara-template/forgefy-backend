# ─────────────────────────────────────────────────────────────────────────────
# Multi-stage build: compile a Flutter WEB app, then serve the static output.
#
# Stage 1 (build):   installs Flutter, runs `flutter build web --release`, and
#                    deletes the SDK engine/caches in the SAME RUN layer so the
#                    multi-GB SDK never persists into an image layer. This keeps
#                    Cloudflare Workers Builds disk usage far below the 20 GB limit.
#
# Stage 2 (runtime): nginx serving the compiled `build/web` directory. The Flutter
#                    SDK is NOT present in the final image.
# ─────────────────────────────────────────────────────────────────────────────

# ================================================================
# Stage 1 — build
# ================================================================
FROM ubuntu:22.04 AS build

ENV DEBIAN_FRONTEND=noninteractive

# Minimal toolchain to download and extract the Flutter SDK. A web build needs no
# clang/cmake/GTK dev headers, so we leave them out to keep this stage small.
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    git \
    unzip \
    xz-utils \
    && rm -rf /var/lib/apt/lists/*

# ── Flutter SDK (pin version — bump to upgrade) ──────────────────────────────
ARG FLUTTER_VERSION=3.24.4
ENV FLUTTER_HOME=/opt/flutter
ENV PATH=$PATH:$FLUTTER_HOME/bin
ENV PUB_CACHE=/root/.pub-cache

# Directory (relative to the build context) that contains the Flutter app's
# pubspec.yaml. Override with:  --build-arg FLUTTER_APP_DIR=path/to/app
ARG FLUTTER_APP_DIR=.

WORKDIR /src
COPY . /src

# Download the SDK, build the web bundle, then purge SDK/cache artifacts in the
# same layer so they are not baked into the layer set.
RUN curl -o /tmp/flutter.tar.xz \
      "https://storage.googleapis.com/flutter_infra_release/releases/stable/linux/flutter_linux_${FLUTTER_VERSION}-stable.tar.xz" && \
    tar xf /tmp/flutter.tar.xz -C /opt && \
    rm /tmp/flutter.tar.xz && \
    git config --global --add safe.directory $FLUTTER_HOME && \
    flutter config --no-analytics && \
    cd ${FLUTTER_APP_DIR:-.} && \
    flutter pub get && \
    flutter build web --release && \
    # Final image only carries build/web; strip the SDK & caches now.
    rm -rf $FLUTTER_HOME/bin/cache \
           $FLUTTER_HOME/.git \
           /root/.pub-cache \
           /tmp/* \
           /var/lib/apt/lists/* \
           /var/cache/apt/archives/*

# ================================================================
# Stage 2 — runtime (small static server)
# ================================================================
FROM nginx:1.27-alpine AS runtime

# Copy ONLY the compiled web bundle from the build stage.
COPY --from=build /src/build/web /usr/share/nginx/html

EXPOSE 80