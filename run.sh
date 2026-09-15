#!/bin/bash

set -a
source .env
set +a

APP_NAME="news-similarity"

echo "Stopping existing $APP_NAME process..."
pm2 delete $APP_NAME 2>/dev/null || true

echo "Starting $APP_NAME on port $PORT..."
pm2 start "uv run python main.py" --name $APP_NAME

pm2 save
pm2 status
