#!/usr/bin/env bash
# Deploy the redaction API to Google Cloud Run (the UI is a Render static site; see render.yaml).
#
#   deploy/deploy_cloud_run.sh https://<your-ui>.onrender.com
#
# Builds the same image as deploy/hf-space (all models baked in) with Cloud Build, then deploys it.
# Settings, and why:
#   --max-instances 1      jobs and uploaded files live in one container's memory; a 2nd instance would 404 polls
#   --no-cpu-throttling    jobs run in background threads after the POST returns; throttled CPU would stall them
#   --cpu 2 --memory 8Gi   peak use measured at 3.6 GB; uploads/outputs are written to an in-memory filesystem
#   --min-instances 0      scale to zero when idle (no cost), at the price of a slow first request
set -euo pipefail

CORS=${1:?usage: deploy/deploy_cloud_run.sh <ui-origin[,more-origins]>}
PROJECT=${PROJECT:-$(gcloud config get-value project 2>/dev/null)}
REGION=${REGION:-us-central1}
SERVICE=${SERVICE:-redact-api}
REPO=${REPO:-redact}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
IMAGE="$REGION-docker.pkg.dev/$PROJECT/$REPO/$SERVICE:$(git -C "$ROOT" rev-parse --short HEAD)"

gcloud services enable run.googleapis.com artifactregistry.googleapis.com cloudbuild.googleapis.com --project "$PROJECT"
gcloud artifacts repositories describe "$REPO" --location "$REGION" --project "$PROJECT" >/dev/null 2>&1 \
  || gcloud artifacts repositories create "$REPO" --repository-format docker --location "$REGION" --project "$PROJECT"

STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT
"$ROOT/.venv/bin/python" "$ROOT/deploy/push_hf_space.py" --stage-only "$STAGE" >/dev/null

# Default build machine; the model downloads make this a long build, hence the 1 h timeout.
gcloud builds submit "$STAGE" --tag "$IMAGE" --timeout 3600s --project "$PROJECT"

gcloud run deploy "$SERVICE" --image "$IMAGE" --region "$REGION" --project "$PROJECT" \
  --port 7860 --cpu 2 --memory 8Gi --no-cpu-throttling --cpu-boost \
  --min-instances 0 --max-instances 1 --concurrency 80 --timeout 900 \
  --allow-unauthenticated \
  --set-env-vars "^@^CORS_ORIGINS=$CORS@ALLOW_ORIGINAL_PLAYBACK=1"

gcloud run services describe "$SERVICE" --region "$REGION" --project "$PROJECT" --format 'value(status.url)'
