#!/usr/bin/env bash
MSG=${1:-init}
echo "Creating alembic revision with message: $MSG"
alembic -c migrations/alembic.ini revision --autogenerate -m "$MSG"
