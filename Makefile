# Makefile for ReAct Integral Agent
# Professional development workflow automation

# Variables for common commands and messages
DOCKER_COMPOSE := docker-compose
APP_CONTAINER := app
DB_CONTAINER := postgres
DB_USER := agent_user
DB_NAME := react_agent_db
POETRY_CMD := $(DOCKER_COMPOSE) run --rm $(APP_CONTAINER) poetry run

# PHONY targets
.PHONY: help setup install clean test lint format quality health info
.PHONY: build up down restart logs logs-live status monitor
.PHONY: dev dev-with-tests dev-quiet db-shell db-reset backup-db 
.PHONY: notebook-test notebook-clean notebook-status
.PHONY: shell-app shell-db poetry-add poetry-remove

# Default target
help: ## Show this help message
	@echo "🤖 ReAct Integral Agent - Development Commands"
	@echo "=============================================="
	@echo ""
	@echo "📊 Environment:"
	@echo "  Branch: $$(git branch --show-current 2>/dev/null || echo 'Not a git repo')"
	@echo "  Docker: $$(docker --version 2>/dev/null || echo 'Not installed')"
	@echo ""
	@echo "🎯 Main Commands:"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# Core workflow commands
setup: check-deps build ## Complete project setup
	@echo "🎉 Setup complete! Use 'make dev' to start development"

dev: up ## Start development environment
	@echo "🚀 Development environment started!"
	@echo "📱 Streamlit: http://localhost:8501"
	@echo "📝 Jupyter: http://localhost:8888"
	@echo "🐘 Database: localhost:5432"
	@echo ""
	@echo "📋 Use 'make logs' to see live service logs"
	@echo "📋 Use 'make test' to run tests separately"

dev-with-tests: up test ## Start development environment and run tests
	@echo "🚀 Development environment started with tests!"
	@echo "📱 Streamlit: http://localhost:8501"
	@echo "📝 Jupyter: http://localhost:8888"
	@echo "🐘 Database: localhost:5432"

dev-quiet: ## Start development environment without service logs
	@echo "🚀 Starting services in quiet mode..."
	@SHOW_LOGS=false $(DOCKER_COMPOSE) up -d
	@echo "🚀 Development environment started (quiet mode)!"
	@echo "📱 Streamlit: http://localhost:8501"
	@echo "📝 Jupyter: http://localhost:8888"
	@echo "🐘 Database: localhost:5432"
	@echo "📋 Use 'make logs-live' to see service logs"

# Dependency checks (DRY principle)
check-deps: ## Check if required tools are installed
	@echo "🔍 Checking dependencies..."
	@command -v docker >/dev/null 2>&1 || { echo "❌ Docker required but not installed."; exit 1; }
	@command -v docker-compose >/dev/null 2>&1 || { echo "❌ Docker Compose required but not installed."; exit 1; }
	@docker info >/dev/null 2>&1 || { echo "❌ Docker daemon not running. Start Docker Desktop."; exit 1; }
	@echo "✅ Docker environment ready"

# Docker operations (modular and reusable)
build: ## Build Docker containers
	@echo "🔨 Building containers..."
	@$(DOCKER_COMPOSE) build --no-cache

up: ## Start all services
	@echo "🚀 Starting services..."
	@$(DOCKER_COMPOSE) up -d

down: ## Stop all services
	@echo "🛑 Stopping services..."
	@$(DOCKER_COMPOSE) down

restart: down up ## Restart all services

logs: ## Show logs from all services
	@echo "📋 Service logs (Ctrl+C to exit):"
	@$(DOCKER_COMPOSE) logs -f

logs-app: ## Show app logs only
	@$(DOCKER_COMPOSE) logs -f $(APP_CONTAINER)

logs-db: ## Show database logs only  
	@$(DOCKER_COMPOSE) logs -f $(DB_CONTAINER)

logs-live: ## Show live service logs (Jupyter & Streamlit)
	@echo "📋 Live service logs (Ctrl+C to exit):"
	@echo "🔍 Tailing Jupyter and Streamlit logs..."
	@$(DOCKER_COMPOSE) exec $(APP_CONTAINER) tail -f /var/log/jupyter.log /var/log/streamlit.log 2>/dev/null || echo "📝 Services are showing logs in console mode"

status: ## Show service status
	@echo "📊 Service Status:"
	@$(DOCKER_COMPOSE) ps

health: ## Check service health
	@echo "🏥 Health Check:"
	@$(DOCKER_COMPOSE) exec $(DB_CONTAINER) pg_isready -U $(DB_USER) -d $(DB_NAME) >/dev/null 2>&1 && echo "✅ Database: Ready" || echo "❌ Database: Down"
	@$(DOCKER_COMPOSE) ps $(APP_CONTAINER) | grep -q "Up" && echo "✅ App: Running" || echo "❌ App: Down"

monitor: ## Monitor resource usage
	@echo "📊 Resource monitoring:"
	@docker stats

# Database operations (single responsibility)
db-shell: ## Connect to database shell
	@echo "🐘 Connecting to database..."
	@$(DOCKER_COMPOSE) exec $(DB_CONTAINER) psql -U $(DB_USER) -d $(DB_NAME)

db-reset: ## Reset database (WARNING: Deletes all data!)
	@echo "⚠️  This will delete ALL database data!"
	@read -p "Continue? [y/N]: " confirm && [ "$$confirm" = "y" ] || exit 1
	@$(DOCKER_COMPOSE) down -v
	@$(DOCKER_COMPOSE) up -d $(DB_CONTAINER)
	@echo "✅ Database reset complete"

backup-db: ## Create database backup
	@echo "💾 Creating backup..."
	@mkdir -p backups
	@$(DOCKER_COMPOSE) exec $(DB_CONTAINER) pg_dump -U $(DB_USER) $(DB_NAME) > backups/backup_$(shell date +%Y%m%d_%H%M%S).sql
	@echo "✅ Backup saved in backups/"

# Testing and Quality (consolidated)
test: ## Run all tests
	@echo "🧪 Running tests..."
	@$(POETRY_CMD) pytest tests/ -v --cov=app --cov-report=term-missing

test-unit: ## Run unit tests only
	@echo "🧪 Running unit tests..."
	@$(POETRY_CMD) pytest tests/unit/ -v

test-integration: ## Run integration tests only
	@echo "🧪 Running integration tests..."
	@$(POETRY_CMD) pytest tests/integration/ -v

test-api-protection: ## Comprehensive API protection and mock infrastructure validation
	@echo "🔒 Running comprehensive API protection validation..."
	@$(DOCKER_COMPOSE) exec $(APP_CONTAINER) python -m tests.integration.test_api_protection

test-api-protection-pytest: ## Run API protection tests with pytest
	@echo "🔒 Running API protection tests with pytest..."
	@$(POETRY_CMD) pytest tests/integration/test_api_protection.py -v -s

lint: ## Run linting checks
	@echo "🔍 Running linters..."
	@$(POETRY_CMD) flake8 app/ tests/ --max-line-length=88 --extend-ignore=E203,W503
	@$(POETRY_CMD) mypy app/ --ignore-missing-imports

format: ## Format code
	@echo "🎨 Formatting code..."
	@$(POETRY_CMD) black app/ tests/
	@$(POETRY_CMD) isort app/ tests/

format-check: ## Check code formatting
	@echo "🎨 Checking formatting..."
	@$(POETRY_CMD) black --check app/ tests/
	@$(POETRY_CMD) isort --check-only app/ tests/

quality: lint format-check test ## Run all quality checks

# Notebook operations (DRY principle applied)
notebook-test: ## Test all notebooks
	@echo "🧪 Testing notebooks..."
	@for nb in notebooks/*.ipynb; do \
		echo "Testing $$(basename $$nb)..."; \
		$(POETRY_CMD) jupyter nbconvert --to notebook --execute --inplace "$$nb" || exit 1; \
	done
	@echo "✅ All notebooks tested"

notebook-clean: ## Clean notebook outputs
	@echo "� Cleaning notebooks..."
	@for nb in notebooks/*.ipynb; do \
		echo "Cleaning $$(basename $$nb)..."; \
		$(DOCKER_COMPOSE) exec $(APP_CONTAINER) python3 scripts/clean_notebook.py "$$nb"; \
	done
	@echo "✅ Notebooks cleaned"

notebook-status: ## Check notebook git status
	@echo "📊 Notebook Status:"
	@for nb in notebooks/*.ipynb; do \
		if git diff --quiet "$$nb" 2>/dev/null; then \
			echo "  ✅ $$(basename $$nb) - Clean"; \
		else \
			echo "  � $$(basename $$nb) - Modified"; \
		fi \
	done

# Container access (productivity shortcuts)
shell-app: ## Shell into app container
	@$(DOCKER_COMPOSE) exec $(APP_CONTAINER) /bin/bash

shell-db: ## Shell into database container
	@$(DOCKER_COMPOSE) exec $(DB_CONTAINER) /bin/bash

# Poetry package management (parameterized)
poetry-add: ## Add package (usage: make poetry-add PACKAGE=package_name)
ifndef PACKAGE
	@echo "❌ Usage: make poetry-add PACKAGE=package_name"
	@exit 1
endif
	@$(DOCKER_COMPOSE) exec $(APP_CONTAINER) poetry add $(PACKAGE)

poetry-remove: ## Remove package (usage: make poetry-remove PACKAGE=package_name)  
ifndef PACKAGE
	@echo "❌ Usage: make poetry-remove PACKAGE=package_name"
	@exit 1
endif
	@$(DOCKER_COMPOSE) exec $(APP_CONTAINER) poetry remove $(PACKAGE)

# Cleanup operations (modular)
clean: ## Clean containers and volumes
	@echo "🧹 Cleaning up..."
	@$(DOCKER_COMPOSE) down -v --remove-orphans
	@docker system prune -f

clean-all: clean ## Deep clean (includes images)
	@echo "🧹 Deep cleaning..."
	@$(DOCKER_COMPOSE) down -v --remove-orphans --rmi all
	@docker system prune -af

# Information and documentation
info: ## Show environment information
	@echo "ℹ️  Environment Information:"
	@echo "Docker: $$(docker --version)"
	@echo "Docker Compose: $$(docker-compose --version)"
	@echo "Directory: $$(pwd)"
	@echo "Git branch: $$(git branch --show-current 2>/dev/null || echo 'Not a git repo')"

# Convenience aliases
install: build ## Alias for build
quick-start: dev ## Alias for dev
