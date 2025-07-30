# Makefile for ReAct Integral Agent
# Professional development workflow automation

.PHONY: help install build up down logs test lint format clean check-deps health status

# Default target
help: ## Show this help message
	@echo "🤖 ReAct Integral Agent - Development Commands"
	@echo "=============================================="
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# Environment setup
install: build ## Install dependencies via Docker (Poetry inside container)
	@echo "📦 Dependencies installed via Docker build (includes Poetry)"
	@echo "✅ Ready for development"

check-deps: ## Check if required tools are installed
	@echo "🔍 Checking dependencies..."
	@command -v docker >/dev/null 2>&1 || { echo "❌ Docker is required but not installed."; exit 1; }
	@command -v docker-compose >/dev/null 2>&1 || { echo "❌ Docker Compose is required but not installed."; exit 1; }
	@docker info >/dev/null 2>&1 || { echo "❌ Docker daemon is not running. Please start Docker Desktop."; exit 1; }
	@echo "✅ Docker is ready (Poetry included in containers)"

# Docker operations
build: ## Build Docker containers
	@echo "🔨 Building Docker containers..."
	docker-compose build --no-cache

up: ## Start all services in detached mode
	@echo "🚀 Starting services..."
	docker-compose up -d

up-logs: ## Start all services with logs
	@echo "🚀 Starting services with logs..."
	docker-compose up

down: ## Stop all services
	@echo "🛑 Stopping services..."
	docker-compose down

restart: down up ## Restart all services

logs: ## Show logs from all services
	@echo "📋 Showing logs..."
	docker-compose logs -f

logs-app: ## Show only app logs
	@echo "📋 Showing app logs..."
	docker-compose logs -f app

logs-db: ## Show only database logs
	@echo "📋 Showing database logs..."
	docker-compose logs -f postgres

# Development
dev: up-logs ## Start development environment with logs

run-local: ## Run app in Docker container
	@echo "🏃 Running app in Docker..."
	docker-compose up app

# Database operations
db-shell: ## Connect to PostgreSQL shell
	@echo "🐘 Connecting to PostgreSQL..."
	docker-compose exec postgres psql -U agent_user -d react_agent_db

db-reset: ## Reset database (WARNING: This will delete all data!)
	@echo "⚠️  Resetting database..."
	@read -p "Are you sure? This will delete all data [y/N]: " confirm && [ "$$confirm" = "y" ]
	docker-compose down -v
	docker-compose up -d postgres
	@echo "✅ Database reset complete"

# Health checks
health: ## Check health of all services
	@echo "🏥 Checking service health..."
	@echo "Database:"
	@docker-compose exec postgres pg_isready -U agent_user -d react_agent_db || echo "❌ Database not ready"
	@echo "App container:"
	@docker-compose ps app | grep -q "Up" && echo "✅ App container running" || echo "❌ App container not running"

status: ## Show status of all services
	@echo "📊 Service Status:"
	docker-compose ps

# Testing and Quality
test: ## Run all tests in Docker
	@echo "🧪 Running tests in Docker (with Poetry)..."
	docker-compose run --rm app poetry run pytest tests/ -v --cov=app --cov-report=term-missing

test-unit: ## Run only unit tests in Docker
	@echo "🧪 Running unit tests in Docker..."
	docker-compose run --rm app poetry run pytest tests/unit/ -v

lint: ## Run linting checks in Docker
	@echo "🔍 Running linting in Docker..."
	docker-compose run --rm app poetry run flake8 app/ tests/ --max-line-length=88 --extend-ignore=E203,W503
	docker-compose run --rm app poetry run mypy app/ --ignore-missing-imports

format: ## Format code with black and isort in Docker
	@echo "🎨 Formatting code in Docker..."
	docker-compose run --rm app poetry run black app/ tests/
	docker-compose run --rm app poetry run isort app/ tests/

format-check: ## Check code formatting in Docker
	@echo "🎨 Checking code formatting in Docker..."
	docker-compose run --rm app poetry run black --check app/ tests/
	docker-compose run --rm app poetry run isort --check-only app/ tests/

quality: lint format-check test ## Run all quality checks

# Git operations
commit-check: quality ## Run quality checks before commit
	@echo "✅ Ready to commit"

# Cleanup
clean: ## Clean up containers, volumes, and cache
	@echo "🧹 Cleaning up..."
	docker-compose down -v --remove-orphans
	docker system prune -f
	poetry cache clear pypi --all

clean-all: clean ## Clean everything including images
	@echo "🧹 Deep cleaning..."
	docker-compose down -v --remove-orphans --rmi all
	docker system prune -af

# Backup and restore
backup-db: ## Backup database
	@echo "💾 Creating database backup..."
	mkdir -p backups
	docker-compose exec postgres pg_dump -U agent_user react_agent_db > backups/backup_$(shell date +%Y%m%d_%H%M%S).sql
	@echo "✅ Backup created in backups/ directory"

# Production helpers
build-prod: ## Build production containers
	@echo "🏭 Building production containers..."
	docker-compose -f docker-compose.yml -f docker-compose.prod.yml build

# Monitoring
monitor: ## Monitor resource usage
	@echo "📊 Monitoring containers..."
	docker stats

# Development workflow shortcuts
setup: check-deps install ## Complete Docker-based setup
	@echo "🎉 Docker setup complete! Run 'make dev' to start development"

quick-start: up ## Quick start (assumes everything is built)
	@echo "🚀 Services started!"
	@echo "📱 App: http://localhost:8501"
	@echo "🐘 Database: localhost:5432"
	@echo "📋 Run 'make logs' to see logs"

# Productivity shortcuts
shell-app: ## Shell into app container
	docker-compose exec app /bin/bash

shell-db: ## Shell into database container  
	docker-compose exec postgres /bin/bash

poetry-shell: ## Poetry shell in app container
	docker-compose exec app poetry shell

poetry-add: ## Add package with Poetry (usage: make poetry-add PACKAGE=package_name)
	docker-compose exec app poetry add $(PACKAGE)

poetry-remove: ## Remove package with Poetry (usage: make poetry-remove PACKAGE=package_name)
	docker-compose exec app poetry remove $(PACKAGE)

# Documentation
docs: ## Generate/update documentation
	@echo "📚 Documentation commands:"
	@echo "  - README.md: Project overview"
	@echo "  - IMPLEMENTATION_PLAN.md: Detailed implementation plan"
	@echo "  - Make targets: Run 'make help'"

# Environment info
info: ## Show environment information
	@echo "ℹ️  Environment Information:"
	@echo "Docker version: $$(docker --version)"
	@echo "Docker Compose version: $$(docker-compose --version)"
	@echo "Poetry version: $$(poetry --version)"
	@echo "Python version: $$(python3 --version)"
	@echo "Current directory: $$(pwd)"
	@echo "Git branch: $$(git branch --show-current 2>/dev/null || echo 'Not a git repo')"
