#!/bin/bash

# Stop Celery Workers and Beat Scheduler

echo "Stopping Nadas.fi Celery Workers..."

# Stop worker
if [ -f celery_worker.pid ]; then
    WORKER_PID=$(cat celery_worker.pid)
    echo "Stopping Celery worker (PID: $WORKER_PID)..."
    kill $WORKER_PID 2>/dev/null
    rm celery_worker.pid
fi

# Stop beat scheduler
if [ -f celery_beat.pid ]; then
    BEAT_PID=$(cat celery_beat.pid)
    echo "Stopping Celery beat (PID: $BEAT_PID)..."
    kill $BEAT_PID 2>/dev/null
    rm celery_beat.pid
fi

# Force kill any remaining celery processes
pkill -f "celery.*app.workers.automation_tasks" 2>/dev/null

echo "Celery stopped successfully!"