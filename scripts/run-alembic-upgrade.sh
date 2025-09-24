#!/usr/bin/env bash
echo "Running alembic upgrade head"
alembic -c migrations/alembic.ini upgrade head
