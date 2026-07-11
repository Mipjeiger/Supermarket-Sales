#!/bin/bash
set -e

echo "⏳ Waiting for MLflow tracking server to start inside container..."
python3 -c "
import socket
import time
while True:
    try:
        with socket.create_connection(('mlflow', 5000), timeout=2):
            print('🚀 MLflow port 5000 is open!')
            break
    except OSError:
        print('⏳ ...waiting for MLflow to accept connections...')
        time.sleep(2)
"

echo "⏳ Waiting for Apache Kafka broker to accept traffic routing..."
python3 -c "
import socket
import time
while True:
    try:
        # Pings the internal docker listener port designated in docker-compose
        with socket.create_connection(('kafka', 29092), timeout=2):
            print('📡 Kafka cluster port 29092 is open and operational!')
            break
    except OSError:
        print('⏳ ...waiting for Kafka KRaft broker node to accept socket mapping...')
        time.sleep(2)
"

echo "📢 Dispatching container initialization log telemetry to Slack..."
python3 -c "
import os, json, urllib.request, time
webhook_url = os.getenv('SLACK_WEBHOOK_URL')
if webhook_url:
    payload = {
        'channel': os.getenv('SLACK_CHANNEL', 'supermarket_sales'),
        'attachments': [{
            'color': '#36a64f',
            'text': '🚀 *Deployment Alert:* Supermarket Backend container has successfully passed dependency gates (MLflow & Kafka) and is initializing Uvicorn application workers.',
            'ts': int(time.time())
        }]
    }
    try:
        req = urllib.request.Request(
            webhook_url, 
            data=json.dumps(payload).encode('utf-8'), 
            headers={'Content-Type': 'application/json'}
        )
        urllib.request.urlopen(req, timeout=5)
        print('✅ Initial boot notification delivered to Slack channel.')
    except Exception as e:
        print(f'⚠️ Slack notify gate bypassed: {e}')
"

echo "📦 Executing automated pipeline registry..."
python3 /app/Production/scripts/register_model.py || echo "⚠️ Script warnings handled, proceeding..."

echo "📡 Starting streaming data pipeline from DB postgres to kafka on automated kafka pipeline messages..."
python3 /app/Production/scripts/postgres_to_kafka.py || echo "⚠️ Script warnings handled, proceeding..."

echo "⚡ Starting FastAPI application instance..."
exec uvicorn app.main:app --host 0.0.0.0 --port 8000