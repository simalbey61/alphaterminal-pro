.PHONY: help dev build up down logs shell test clean

# Default
help:
	@echo "AlphaTerminal Pro - Makefile"
	@echo "============================"
	@echo ""
	@echo "Usage:"
	@echo "  make dev        - Start development environment"
	@echo "  make build      - Build all containers"
	@echo "  make up         - Start all services"
	@echo "  make down       - Stop all services"
	@echo "  make logs       - View logs"
	@echo "  make shell      - Open backend shell"
	@echo "  make test       - Run tests"
	@echo "  make clean      - Remove all containers and volumes"
	@echo "  make migrate    - Run database migrations"

# Development
dev:
	docker-compose up -d postgres redis
	cd backend && python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Build
build:
	docker-compose build

# Start services
up:
	docker-compose up -d

# Stop services
down:
	docker-compose down

# View logs
logs:
	docker-compose logs -f

# Backend logs only
logs-backend:
	docker-compose logs -f backend

# Shell access
shell:
	docker-compose exec backend bash

# Run tests
test:
	docker-compose exec backend pytest -v

# Clean everything
clean:
	docker-compose down -v --remove-orphans
	docker system prune -f

# Database migrations
migrate:
	docker-compose exec backend alembic upgrade head

# Create new migration
migration:
	docker-compose exec backend alembic revision --autogenerate -m "$(name)"

# Restart services
restart:
	docker-compose restart

# Status
status:
	docker-compose ps
