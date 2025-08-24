#!/bin/bash

# Start Celery Workers and Beat Scheduler for Nadas.fi Backend

echo "Starting Nadas.fi Celery Workers..."

# Set environment variables
export PYTHONPATH="/Users/selahattinozcan/nadas/backend:$PYTHONPATH"
cd /Users/selahattinozcan/nadas/backend

# Start Celery worker in background
echo "Starting Celery worker..."
python3 -m celery -A app.workers.automation_tasks worker --loglevel=info --concurrency=4 &
WORKER_PID=$!

# Start Celery beat scheduler in background  
echo "Starting Celery beat scheduler..."
python3 -m celery -A app.workers.automation_tasks beat --loglevel=info &
BEAT_PID=$!

echo "Celery worker PID: $WORKER_PID"
echo "Celery beat PID: $BEAT_PID"

# Save PIDs for stopping later
echo $WORKER_PID > celery_worker.pid
echo $BEAT_PID > celery_beat.pid

echo "Celery started successfully!"
echo "To stop: ./stop_celery.sh"