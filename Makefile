.PHONY: help build up down restart logs shell test clean prod-build prod-up prod-down

# Default target
help:
	@echo "Available commands:"
	@echo "  make build        - Build development containers"
	@echo "  make up          - Start development environment"
	@echo "  make down        - Stop development environment"
	@echo "  make restart     - Restart development environment"
	@echo "  make logs        - View logs from all services"
	@echo "  make logs-api    - View logs from API service"
	@echo "  make logs-db     - View logs from MySQL service"
	@echo "  make shell       - Open shell in API container"
	@echo "  make test        - Run tests in container"
	@echo "  make clean       - Remove containers, volumes, and images"
	@echo "  make prod-build  - Build production containers"
	@echo "  make prod-up     - Start production environment"
	@echo "  make prod-down   - Stop production environment"

# Development
build:
	docker-compose build

up:
	docker-compose up -d
	@echo "Services started:"
	@echo "  API:     http://localhost:8000"
	@echo "  Docs:    http://localhost:8000/docs"
	@echo "  MySQL:   localhost:3306"

down:
	docker-compose down

restart: down up

logs:
	docker-compose logs -f

logs-api:
	docker-compose logs -f api

logs-db:
	docker-compose logs -f mysql

shell:
	docker-compose exec api /bin/bash

test:
	docker-compose exec api pytest -v

clean:
	docker-compose down -v
	docker system prune -f

# Production
prod-build:
	docker-compose -f docker-compose.yml -f docker-compose.prod.yml build

prod-up:
	docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

prod-down:
	docker-compose -f docker-compose.yml -f docker-compose.prod.yml down

# Database
db-migrate:
	docker-compose exec api python -c "from app.core.database import init_db; init_db()"

db-shell:
	docker-compose exec mysql mysql -u nb_user -pnb_password naive_bayes
