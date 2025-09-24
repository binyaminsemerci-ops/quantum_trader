Write-Host "Running alembic upgrade head"
alembic -c migrations/alembic.ini upgrade head
