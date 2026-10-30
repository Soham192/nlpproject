#!/bin/sh
set -e
# Optional Space secret REDACT_KEY_B64 (base64 of 32 random bytes): a stable AES key, so outputs masked
# before a restart can still be unmasked. Without it the API generates a fresh key on first use.
if [ -n "$REDACT_KEY_B64" ]; then
  python -c "import base64,os; k=base64.b64decode(os.environ['REDACT_KEY_B64']); assert len(k)==32, 'REDACT_KEY_B64 must decode to 32 bytes'; fd=os.open('key.bin', os.O_WRONLY|os.O_CREAT|os.O_TRUNC, 0o600); os.write(fd,k); os.close(fd)"
fi
# Every model is baked into the image (prefetch.py); don't spend time checking the Hub on each load.
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# Cloud Run injects PORT (deployed with --port 7860); Hugging Face Spaces expect 7860.
exec uvicorn api.main:app --host 0.0.0.0 --port "${PORT:-7860}"
