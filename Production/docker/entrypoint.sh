#!/bin/bash
set -e

echo "⏳ Waiting for MLflow tracking server to start inside container..."

# Use native Python to check the port, eliminating the curl dependency and ensuring compatibility across environments
python3 -c "
import socket
import time
while True:
    try:
        with socket.create_connection(('mlflow', 5000), timeout=2):
            print('🚀 Mlflow port 5000 is open!')
            break
    except OSError:
        print('⏳ ...waiting for MLflow to accept connections...')
        time.sleep(2)
"

echo "📦 Executing automated pipeline registry..."
python3 /app/Production/scripts/register_model.py || echo "⚠️ Script warnings handled, proceeding..."

echo "⚡ Starting FastAPI application instance..."
exec uvicorn app.main:app --host 0.0.0.0 --port 8000