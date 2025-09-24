param(
    [string]$Message = "init"
)

Write-Host "Creating alembic revision with message: $Message"
alembic -c migrations/alembic.ini revision --autogenerate -m $Message
