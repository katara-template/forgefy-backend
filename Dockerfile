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
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*


# ── Node.js 22.x ──────────────────────────────────────────────────────────────
RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && \
    apt-get install -y --no-install-recommends nodejs && \
    rm -rf /var/lib/apt/lists/*


# ── OpenJDK ───────────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    openjdk-21-jdk-headless \
    && rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64


# ── Android SDK ───────────────────────────────────────────────────────────────
ENV ANDROID_HOME=/opt/android-sdk
ENV ANDROID_SDK_ROOT=/opt/android-sdk

ENV PATH=$PATH:$ANDROID_HOME/cmdline-tools/latest/bin:$ANDROID_HOME/platform-tools:$ANDROID_HOME/build-tools/34.0.0


RUN mkdir -p $ANDROID_HOME/cmdline-tools && \
    curl -o /tmp/cmdline-tools.zip \
      "https://dl.google.com/android/repository/commandlinetools-linux-11076708_latest.zip" && \
    unzip -q /tmp/cmdline-tools.zip -d /tmp/ct && \
    mv /tmp/ct/cmdline-tools $ANDROID_HOME/cmdline-tools/latest && \
    rm -rf /tmp/cmdline-tools.zip /tmp/ct


RUN yes | sdkmanager --licenses > /dev/null 2>&1 && \
    sdkmanager \
      "platform-tools" \
      "build-tools;34.0.0" \
      "platforms;android-34"


# ── Flutter SDK ───────────────────────────────────────────────────────────────
ARG FLUTTER_VERSION=3.24.4

ENV FLUTTER_HOME=/opt/flutter
ENV PUB_CACHE=/opt/pub-cache

ENV PATH=$FLUTTER_HOME/bin:$FLUTTER_HOME/bin/cache/dart-sdk/bin:$PATH


RUN curl -o /tmp/flutter.tar.xz \
      "https://storage.googleapis.com/flutter_infra_release/releases/stable/linux/flutter_linux_${FLUTTER_VERSION}-stable.tar.xz" && \
    tar xf /tmp/flutter.tar.xz -C /opt && \
    rm /tmp/flutter.tar.xz && \
    git config --global --add safe.directory /opt/flutter && \
    flutter config --no-analytics && \
    flutter precache --android


# ── Workspace ─────────────────────────────────────────────────────────────────
RUN mkdir -p /workspace /opt/pub-cache


# ── Python dependencies ───────────────────────────────────────────────────────
COPY requirements.txt ./

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt


# ── Playwright ───────────────────────────────────────────────────────────────
RUN playwright install chromium --with-deps


# ── Application ───────────────────────────────────────────────────────────────
COPY . .


# ── Create non-root Forgefy user ──────────────────────────────────────────────
RUN useradd -m -s /bin/bash forgefy && \
    chown -R forgefy:forgefy /workspace /opt/pub-cache /opt/flutter


USER forgefy

WORKDIR /app


# ── Environment ───────────────────────────────────────────────────────────────
ENV HOME=/home/forgefy
ENV PUB_CACHE=/opt/pub-cache


# ── Verify toolchain ──────────────────────────────────────────────────────────
RUN flutter --version && \
    dart --version && \
    node --version && \
    java -version


EXPOSE 8000


HEALTHCHECK --interval=10s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/health || exit 1


# ── Start Forgefy API ─────────────────────────────────────────────────────────
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} \
    --workers ${WEB_CONCURRENCY:-4} \
    --proxy-headers --forwarded-allow-ips="*"