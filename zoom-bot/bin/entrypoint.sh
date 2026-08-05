#!/usr/bin/env bash
#
# Prepares the virtual audio device the Meeting SDK requires, then hands off to
# the Python sidecar — which owns the process tree from that point on and
# spawns the SDK client itself.
set -euo pipefail

setup_pulseaudio() {
  # PulseAudio needs a system bus.
  if [[ ! -S /var/run/dbus/system_bus_socket ]]; then
    mkdir -p /var/run/dbus
    dbus-uuidgen > /var/lib/dbus/machine-id 2>/dev/null || true
    dbus-daemon --config-file=/usr/share/dbus-1/system.conf --fork
  fi

  usermod -aG pulse-access,audio root 2>/dev/null || true

  # Stale state from a previous run stops the daemon coming up, and these
  # containers are recycled per meeting.
  rm -rf /var/run/pulse /var/lib/pulse /root/.config/pulse
  mkdir -p ~/.config/pulse
  cp -r /etc/pulse/* ~/.config/pulse/ 2>/dev/null || true

  pulseaudio -D --exit-idle-time=-1 --system --disallow-exit

  # A null sink gives the SDK somewhere to render playback. We never read from
  # it — meeting audio is captured through the raw data API, not the device.
  pactl load-module module-null-sink sink_name=ForgefySink >/dev/null
  pactl set-default-sink ForgefySink
  pactl set-default-source ForgefySink.monitor

  # Without this the Linux client tries to enumerate hardware devices.
  printf '[General]\nsystem.audio.type=default\n' > ~/.config/zoomus.conf
}

echo "[entrypoint] configuring virtual audio" >&2
if ! setup_pulseaudio >/dev/null 2>&1; then
  echo '{"event":"status","status":"error","detail":"pulseaudio setup failed"}'
  exit 1
fi

mkdir -p "$(dirname "${FORGEFY_AUDIO_SOCKET:-/tmp/forgefy/audio.sock}")"

echo "[entrypoint] starting sidecar" >&2
exec python3 -m sidecar.run
