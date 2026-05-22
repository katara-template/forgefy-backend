# PUBLIC_API_BASE_URL — What It Is, How to Get It, and How to Use It

## Table of Contents

1. [What Is PUBLIC_API_BASE_URL?](#what-is-public_api_base_url)
2. [Why Is It Needed?](#why-is-it-needed)
3. [How the URL Is Used in Code](#how-the-url-is-used-in-code)
4. [Getting Your Public URL with ngrok](#getting-your-public-url-with-ngrok)
   - [Step 1 — Install ngrok](#step-1--install-ngrok)
   - [Step 2 — Create a Free Account](#step-2--create-a-free-account)
   - [Step 3 — Authenticate ngrok on Your Machine](#step-3--authenticate-ngrok-on-your-machine)
   - [Step 4 — Start a Tunnel](#step-4--start-a-tunnel)
   - [Step 5 — Copy Your Public URL](#step-5--copy-your-public-url)
5. [Setting the Variable in .env](#setting-the-variable-in-env)
6. [Verifying the Webhook Is Reachable](#verifying-the-webhook-is-reachable)
7. [Updating the URL Each Session](#updating-the-url-each-session)
8. [Production Setup (Non-ngrok)](#production-setup-non-ngrok)
9. [Troubleshooting](#troubleshooting)

---

## What Is PUBLIC_API_BASE_URL?

`PUBLIC_API_BASE_URL` is the **base domain of your API server as seen from the public internet**.

It is the part of the URL before any path. For example:

```
https://a1b2-203-0-113-42.ngrok-free.app
```

It does **not** include a trailing slash or any path like `/api/v1/...`. The application code adds the correct path automatically.

---

## Why Is It Needed?

When you start a Google Meet, Zoom, or Teams session, Forgefy dispatches a **Recall.ai cloud bot** to join the meeting. Recall.ai is a third-party service running in the cloud — it is **not** on your machine.

After the bot joins the meeting it needs to send two types of events back to your server:

| Event | What it carries |
|---|---|
| `transcript.data` | Real-time transcript chunks as people speak |
| `bot.status_change` | Bot lifecycle — joining, recording started, call ended |

Recall.ai delivers these events by making **HTTP POST requests** (webhooks) to your server. For this to work, your server must be reachable from Recall's cloud servers. When you run the backend locally on `localhost:5000`, Recall cannot reach it because `localhost` is only accessible from your own computer.

`PUBLIC_API_BASE_URL` tells the Recall.ai bot exactly where to send those webhooks.

---

## How the URL Is Used in Code

When `dispatch_connector` is called (via Celery) it builds a `RecallConnector` and calls `.join()`:

**`app/connectors/recall.py` line 32**
```python
self._webhook_url = webhook_base_url.rstrip("/") + "/api/v1/webhooks/recall"
```

So if your `.env` has:
```
PUBLIC_API_BASE_URL=https://a1b2-203-0-113-42.ngrok-free.app
```

The full webhook URL sent to Recall becomes:
```
https://a1b2-203-0-113-42.ngrok-free.app/api/v1/webhooks/recall
```

Recall.ai POSTs all transcript and status events to that exact URL.

The handler at `app/api/v1/webhooks.py` (`POST /api/v1/webhooks/recall`) receives these events and:
- Publishes transcript text to Redis → WebSocket → browser
- Enqueues LangGraph feature extraction
- Transitions session state (JOINING → LISTENING → PROCESSING)
- Triggers blueprint generation when the call ends

Without `PUBLIC_API_BASE_URL` set, the webhook URL is empty or wrong and Recall cannot deliver any events. Your session will stay in `JOINING` state forever and no transcript will arrive.

---

## Getting Your Public URL with ngrok

ngrok is a free tool that creates a secure tunnel from the internet to your local machine. It gives you a real `https://` URL that forwards to `localhost:5000`.

### Step 1 — Install ngrok

**Windows (using winget):**
```powershell
winget install ngrok.ngrok
```

**Windows (manual download):**
1. Go to [https://ngrok.com/download](https://ngrok.com/download)
2. Click **Windows** and download the `.zip` file
3. Unzip it — you get a single `ngrok.exe`
4. Move `ngrok.exe` to a folder that is on your PATH, for example `C:\Windows\System32`, or just keep it in your project folder and run it from there

**macOS:**
```bash
brew install ngrok/ngrok/ngrok
```

**Linux:**
```bash
curl -sSL https://ngrok-agent.s3.amazonaws.com/ngrok.asc \
  | sudo tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null \
  && echo "deb https://ngrok-agent.s3.amazonaws.com buster main" \
  | sudo tee /etc/apt/sources.list.d/ngrok.list \
  && sudo apt update && sudo apt install ngrok
```

Verify the install:
```bash
ngrok version
# ngrok version 3.x.x
```

---

### Step 2 — Create a Free Account

1. Go to **[https://dashboard.ngrok.com/signup](https://dashboard.ngrok.com/signup)**
2. Sign up with email, Google, or GitHub — it's free
3. After signing in you land on the ngrok dashboard

The free tier gives you:
- 1 active tunnel at a time
- Random subdomain (changes each restart)
- HTTPS included automatically
- No time limit per session

---

### Step 3 — Authenticate ngrok on Your Machine

On the ngrok dashboard, go to **Getting Started → Your Authtoken** (left sidebar).

You will see a command like:
```bash
ngrok config add-authtoken 2abc1234XYZ_yourTokenHere
```

Run that command in your terminal. This writes your token to `~/.config/ngrok/ngrok.yml` (or `%APPDATA%\ngrok\ngrok.yml` on Windows). You only need to do this once per machine.

---

### Step 4 — Start a Tunnel

Make sure your FastAPI backend is already running:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Then, in a **separate terminal**, start ngrok pointing at port 8000:
```bash
ngrok http 8000
```

You will see output similar to this:
```
ngrok

Session Status                online
Account                       your@email.com (Plan: Free)
Version                       3.x.x
Region                        United States (us)
Latency                       23ms
Web Interface                 http://127.0.0.1:4040
Forwarding                    https://a1b2-203-0-113-42.ngrok-free.app -> http://localhost:5000

Connections                   ttl     opn     rt1     rt5     p50     p90
                              0       0       0.00    0.00    0.00    0.00
```

Keep this terminal open. If you close it, the tunnel dies and Recall webhooks stop working immediately.

---

### Step 5 — Copy Your Public URL

From the ngrok output, copy the `Forwarding` URL — the one starting with `https://`:

```
https://a1b2-203-0-113-42.ngrok-free.app
```

This is your `PUBLIC_API_BASE_URL`. Copy it exactly as shown, **without** a trailing slash.

---

## Setting the Variable in .env

Open `forgefy-backend/.env` and set:

```dotenv
PUBLIC_API_BASE_URL=https://a1b2-203-0-113-42.ngrok-free.app
```

Replace `a1b2-203-0-113-42.ngrok-free.app` with whatever subdomain ngrok gave you.

After editing `.env`, restart the backend and Celery worker so they pick up the new value:

```bash
# Stop uvicorn (Ctrl+C) then restart
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Stop Celery (Ctrl+C) then restart
celery -A app.workers.celery_app worker --queues=meeting.audio,meeting.transcribe,meeting.extract --loglevel=info
```

> **Important:** The Celery worker reads settings at startup. If you change `.env` while the worker is running you must restart it.

---

## Verifying the Webhook Is Reachable

Before testing a real meeting, verify your webhook URL responds:

```bash
curl -s -o /dev/null -w "%{http_code}" \
  -X POST https://a1b2-203-0-113-42.ngrok-free.app/api/v1/webhooks/recall \
  -H "Content-Type: application/json" \
  -d '{"event":"ping","data":{}}'
```

You should get back `204` (no content — the endpoint ignores unknown event types). Any `5xx` means the backend is not running correctly. A `000` or connection error means ngrok is not running or the URL is wrong.

You can also watch live requests in the **ngrok web inspector** at [http://127.0.0.1:4040](http://127.0.0.1:4040) — every webhook Recall sends will show up there in real time, including full request/response bodies.

---

## Updating the URL Each Session

**On the free ngrok plan, your subdomain changes every time you restart ngrok.**

This means every development session you need to:

1. Start ngrok: `ngrok http 8000`
2. Copy the new `https://` URL from the terminal output
3. Update `PUBLIC_API_BASE_URL` in `.env`
4. Restart the Celery worker

To avoid this friction on a paid ngrok plan ($10/month) you can reserve a **static domain**:
1. Go to [https://dashboard.ngrok.com/domains](https://dashboard.ngrok.com/domains)
2. Click **New Domain** — you get a permanent subdomain like `yourname.ngrok.app`
3. Start ngrok with it: `ngrok http --domain=yourname.ngrok.app 8000`
4. Set `PUBLIC_API_BASE_URL=https://yourname.ngrok.app` once and never change it again

---

## Production Setup (Non-ngrok)

When deploying to a real server (VPS, AWS, Railway, Render, etc.) you already have a permanent public domain. Set:

```dotenv
PUBLIC_API_BASE_URL=https://api.yourdomain.com
```

Make sure:
- Port 443 (HTTPS) is open and the SSL certificate is valid — Recall.ai only delivers webhooks to HTTPS URLs
- Recall.ai's IP ranges are not blocked by your firewall (they do not publish a fixed range; if you restrict inbound traffic, allow all sources on port 443)
- `RECALL_WORKSPACE_VERIFICATION_SECRET` is set in both your `.env` and in the Recall dashboard so webhook requests are authenticated

To configure the secret in Recall:
1. Log in at [https://api.recall.ai](https://api.recall.ai) → **Settings → Workspace**
2. Find **Webhook Verification Secret** and generate one
3. Copy the value into your `.env`:
   ```dotenv
   RECALL_WORKSPACE_VERIFICATION_SECRET=the_secret_value_from_dashboard
   ```

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Session stays in `JOINING` forever | `PUBLIC_API_BASE_URL` is empty or wrong | Set the correct ngrok URL in `.env` and restart the worker |
| No transcript appears in the browser | Same as above OR Redis is not running | Check ngrok is active; check `redis-cli ping` returns `PONG` |
| `401 Invalid webhook secret` in logs | `RECALL_WORKSPACE_VERIFICATION_SECRET` mismatch | Match the value in `.env` to the one in the Recall dashboard |
| Webhook POSTs arrive but session is not found | Redis mapping expired or Celery did not store it | Check Celery worker logs for `Recall bot created`; ensure Redis is running with enough memory |
| ngrok shows `ERR_NGROK_3200` (tunnel not found) | You closed the ngrok terminal | Restart `ngrok http 8000` and update `PUBLIC_API_BASE_URL` |
| `curl` returns `000` | ngrok is not running | Start `ngrok http 8000` first |
| Bot joins the call but leaves immediately | `everyone_left_timeout` fired before host joined | This is normal if you tested with an empty meeting; join the meeting first then start the session |
