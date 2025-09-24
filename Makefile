.PHONY: dev-check create-db lint typecheck test

dev-check: create-db lint typecheck test

create-db:
	python -c "from backend.database import Base, engine; Base.metadata.create_all(bind=engine)"

lint:
	ruff check backend

typecheck:
	mypy backend --exclude 'backend/tests/.*'

test:
	pytest -q backend/tests
