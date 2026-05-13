.PHONY: up down dev migrate seed test health

# Docker
up:
	docker compose up -d

down:
	docker compose down

logs:
	docker compose logs -f backend

# Local development
dev:
	PYTHONPATH=. uvicorn backend.main:app --host 0.0.0.0 --port 9000 --reload

# Database
migrate:
	PYTHONPATH=. python -m backend.db.migrate

seed:
	PYTHONPATH=. python scripts/seed_config.py

migrate-duckdb:
	PYTHONPATH=. python scripts/migrate_duckdb_to_sqlite.py

migrate-csv:
	PYTHONPATH=. python scripts/migrate_csv_to_duckdb.py

# Health check
health:
	curl -s http://localhost:9000/health | python -m json.tool

# Run pipeline
run:
	curl -s -X POST http://localhost:9000/api/agent/run \
		-H "Content-Type: application/json" \
		-d '{"max_posts": 5}' | python -m json.tool

# Frontend
frontend-dev:
	cd frontend && npm run dev

frontend-install:
	cd frontend && npm install
