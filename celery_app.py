#!/usr/bin/env python3
"""
Celery App - Production Configuration
Entry point for Celery workers and beat scheduler
"""

from app.workers.automation_tasks import celery_app

# Import all tasks to register them
from app.workers import automation_tasks, notification_tasks, monitoring_tasks

if __name__ == '__main__':
    celery_app.start()