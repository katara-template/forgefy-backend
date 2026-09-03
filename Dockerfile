# ─────────────────────────────────────────────────────────────────────────────
# Multi-stage build to keep the final image small.
#
# Stage 1 (build):   installs Flutter + Linux toolchain, compiles the Flutter
#                    app in release mode, then deletes Flutter's engine caches
#                    in the SAME RUN layer so the ~9 GB of SDK/engine artifacts
#                    never get persisted into an intermediate image layer.
#                    This keeps the Cloudflare Workers Builds disk usage well
#                    below the 20 GB limit.
#
# Stage 2 (runtime): minimal Ubuntu image with ONLY the shared libraries a
#                    Flutter Linux (GTK) app needs at runtime. The Flutter SDK
#                    does NOT appear in the final image.
# ─────────────────────────────────────────────────────────────────────────────

# ================================================================
# Stage 1 — build
# ================================================================
FROM ubuntu:22.04 AS build

ENV DEBIAN_FRONTEND=noninteractive

# Toolchain needed to extract Flutter and to compile the Linux app.
# (Flutter's official Ubuntu prerequisites: clang cmake ninja-build pkg-config
#  libgtk-3-dev liblzma-dev libstdc++-12-dev)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    unzip \
    xz-utils \
    clang \
    cmake \
    ninja-build \
    pkg-config \
    libgtk-3-dev \
    liblzma-dev \
    libstdc++-12-dev \
    && rm -rf /var/lib/apt/lists/*

# ── Flutter SDK (pin version — bump to upgrade) ──────────────────────────────
ARG FLUTTER_VERSION=3.24.4
ENV FLUTTER_HOME=/opt/flutter
ENV PATH=$PATH:$FLUTTER_HOME/bin
ENV PUB_CACHE=/root/.pub-cache

COPY . .

WORKDIR /app

# Extract SDK, build release bundle, then purge Flutter engine/cache artifacts
# in the same layer so they are not baked into the layer set.
RUN curl -o /tmp/flutter.tar.xz \
      "https://storage.googleapis.com/flutter_infra_release/releases/stable/linux/flutter_linux_${FLUTTER_VERSION}-stable.tar.xz" && \
    tar xf /tmp/flutter.tar.xz -C /opt && \
    rm /tmp/flutter.tar.xz && \
    git config --global --add safe.directory $FLUTTER_HOME && \
    flutter config --no-analytics && \
    flutter build linux --release && \
    # Remove SDK engine artifacts, pub cache, temp files, and apt caches now.
    rm -rf $FLUTTER_HOME/bin/cache \
           $FLUTTER_HOME/.git \
           /root/.pub-cache \
           /tmp/* \
           /var/lib/apt/lists/* \
           /var/cache/apt/archives/*

# ================================================================
# Stage 2 — runtime (small)
# ================================================================
FROM ubuntu:22.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive

# Shared libs a Flutter Linux/GTK app needs at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgtk-3-0 \
    libglib2.0-0 \
    libblkid1 \
    liblzma5 \
    libc6 \
    libstdc++6 \
    libasound2 \
    libx11-6 \
    libxcursor1 \
    libxdamage1 \
    libxext6 \
    libxfixes3 \
    libxi6 \
    libxrandr2 \
    libxrender1 \
    libxtst6 \
    libwayland-client0 \
    libwayland-cursor0 \
    libfontconfig1 \
    libfreetype6 \
    libpng16-16 \
    libjpeg8 \
    libsqlite3-0 \
    zlib1g \
    xz-utils \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy ONLY the compiled release bundle from the build stage.
COPY --from=build /app/build/linux/x64/release/bundle .

CMD ["./app"]
