#!/usr/bin/env bash
# Phase 22: pull the latest code from GitHub and restart the app.
# On the VPS:  cd /opt/hybrid-rag-app && ./deploy/deploy.sh
set -euo pipefail
cd "$(dirname "$0")/.."

[ -f .env ] || { echo "Missing .env - copy it from your PC first."; exit 1; }
grep -q '^FRONTEND_BIND=127.0.0.1:' .env || { echo "Set BACKEND_BIND/FRONTEND_BIND to 127.0.0.1:... in .env"; exit 1; }

echo "== Pulling latest code =="
git pull --ff-only

# Containers run as uid 1000, so it must own the data folders
mkdir -p data evaluation/results logs
chown -R 1000:1000 data evaluation logs

echo "== Building and starting containers =="
docker compose up -d --build --remove-orphans

echo "== Waiting for the backend to become healthy =="
for i in $(seq 1 40); do
  status=$(docker inspect -f '{{.State.Health.Status}}' hybrid-rag-backend 2>/dev/null || echo starting)
  [ "$status" = healthy ] && break
  sleep 5
done
docker compose up -d                      # starts the frontend once the backend is healthy
docker compose ps
docker image prune -f >/dev/null
echo "== Done: https://rag.palanitech.online =="
