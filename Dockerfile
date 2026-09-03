# ─────────────────────────────────────────────────────────────────────────────
# forgefy-api backend image.
#
# This is a Python (FastAPI/uvicorn) service. It does NOT build a Flutter bundle
# in the image. The heavy mobile/web toolchain below (Node, JDK, Android SDK,
# Flutter) is installed so the BACKEND'S build-agent code can generate user apps
# at RUNTIME. Docker build only sets them up — it never runs `flutter build`.
#
# To keep the Cloudflare 20 GB build disk safe, we:
#   * avoid `flutter precache` / `flutter doctor --android-licenses`, which pull
#     the huge engine/artifact caches that are only needed when building apps,
#   * delete temp archives & apt caches in the same RUN layer as their install.
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.14-slim

WORKDIR /app

# ── Base system tools ─────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    curl \
    git \
    unzip \
    wget \
    xz-utils \
    file \
    && rm -rf /var/lib/apt/lists/*

# ── Node.js 22.x (LTS) ───────────────────────────────────────────────────────
RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && \
    apt-get install -y --no-install-recommends nodejs && \
    rm -rf /var/lib/apt/lists/*

# ── OpenJDK 21 ───────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    openjdk-21-jdk-headless \
    && rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64

# ── Android SDK ───────────────────────────────────────────────────────────────
ENV ANDROID_HOME=/opt/android-sdk
ENV PATH=$PATH:$ANDROID_HOME/cmdline-tools/latest/bin:$ANDROID_HOME/platform-tools:$ANDROID_HOME/build-tools/34.0.0

RUN mkdir -p $ANDROID_HOME/cmdline-tools && \
    curl -o /tmp/cmdline-tools.zip \
      "https://dl.google.com/android/repository/commandlinetools-linux-11076708_latest.zip" && \
    unzip -q /tmp/cmdline-tools.zip -d /tmp/ct && \
    mv /tmp/ct/cmdline-tools $ANDROID_HOME/cmdline-tools/latest && \
    rm -rf /tmp/cmdline-tools.zip /tmp/ct

# Licenses are required even at runtime, but they don't need the SDK's own
# big engines unconditionally — accept them, then install platform tooling.
RUN yes | sdkmanager --licenses > /dev/null 2>&1 && \
    sdkmanager --install \
      "platform-tools" \
      "build-tools;34.0.0" \
      "platforms;android-34"

# ── Flutter SDK (pin version — update ARG to upgrade) ────────────────────────
ARG FLUTTER_VERSION=3.24.4
ENV FLUTTER_HOME=/opt/flutter
ENV PATH=$PATH:$FLUTTER_HOME/bin
ENV PUB_CACHE=/root/.pub-cache

RUN curl -o /tmp/flutter.tar.xz \
      "https://storage.googleapis.com/flutter_infra_release/releases/stable/linux/flutter_linux_${FLUTTER_VERSION}-stable.tar.xz" && \
    tar xf /tmp/flutter.tar.xz -C /opt && \
    rm -rf /tmp/flutter.tar.xz && \
    git config --global --add safe.directory /opt/flutter && \
    flutter config --no-analytics

# ── Python dependencies ───────────────────────────────────────────────────────
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Install Chromium for the Meet connector bot
RUN playwright install chromium --with-deps

# Copy the backend source (sdks/ is excluded via .dockerignore).
COPY . .

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/health || exit 1

# Shell form so the platform can tune via env without an image rebuild:
#   WEB_CONCURRENCY — worker processes (default 4)
#   PORT            — listen port (defaults to 8000)
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} \
    --workers ${WEB_CONCURRENCY:-4} \
    --proxy-headers --forwarded-allow-ips="*"