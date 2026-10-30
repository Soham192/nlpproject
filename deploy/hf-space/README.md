---
title: Selective Audio Redaction API
emoji: 🔒
colorFrom: purple
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

# Selective audio redaction: backend

FastAPI backend for the selective audio redaction demo. The UI is hosted separately on Render and calls
this Space's `/api/*` endpoints.

**Public demo.** Anyone with the link can upload audio, and original playback is enabled. Upload only
synthetic or consented recordings, never real customer data.

Space variables:
- `CORS_ORIGINS`: comma-separated UI origin(s) allowed to call the API, e.g. `https://<your-ui>.onrender.com`
- `ALLOW_ORIGINAL_PLAYBACK`: `1` (demo) or `0`
- optional secret `REDACT_KEY_B64`: a stable AES-256 key (base64 of 32 bytes)
