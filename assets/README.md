# Assets

## Bot avatar (`bot_avatar.jpg`)

Drop a JPEG named `bot_avatar.jpg` in this folder and the Recall meeting bot
will display it as its camera feed when it joins a call — instead of the
platform's default letter avatar.

Requirements:
- **Format:** JPEG only (`.jpg` / `.jpeg`)
- **Aspect ratio:** 16:9 recommended (e.g. 1280×720) so it fills the video tile
- **Size:** ≤ 2 MB (it's sent base64-inlined on every bot join)

If the file is missing the bot joins normally without an avatar — nothing
breaks. The path can be changed via `RECALL_BOT_AVATAR_PATH` in `.env`.
