# Makefile for ReAct Integral Agent
# Professional development workflow automation

.PHONY: help install build up down logs test lint format clean check-deps health status

# Default target
help: ## Show this help message
	@echo "🤖 # Notebook management
notebook-status: ## Check which notebooks have changes
	@echo "📊 Notebook Status	@echo "Git branch: $$(git branch --show-current 2>/dev/null || echo 'Not a git repo')"n notebooks/*.ipynb; do \
		if git diff --quiet "$$notebook" 2>/dev/null; then \
			echo "  ✅ $$(basename $$notebook) - No changes"; \
		else \
			echo "  📝 $$(basename $$notebook) - Has changes"; \
		fi \
	done

track-notebook: ## Enable tracking for a specific notebook (usage: make track-notebook NOTEBOOK=filename.ipynb)
	@if [ -z "$(NOTEBOOK)" ]; then \
		echo "❌ Please specify NOTEBOOK=filename.ipynb"; \
		exit 1; \
	fi
	@echo "📝 Enabling tracking for notebooks/$(NOTEBOOK)..."
	git update-index --no-skip-worktree "notebooks/$(NOTEBOOK)"
	@echo "✅ Now git will detect changes in notebooks/$(NOTEBOOK)"

untrack-notebook: ## Disable tracking for a specific notebook (usage: make untrack-notebook NOTEBOOK=filename.ipynb)
	@if [ -z "$(NOTEBOOK)" ]; then \
		echo "❌ Please specify NOTEBOOK=filename.ipynb"; \
		exit 1; \
	fi
	@echo "🔄 Disabling tracking for notebooks/$(NOTEBOOK)..."
	git update-index --skip-worktree "notebooks/$(NOTEBOOK)"
	@echo "✅ Now git will ignore execution changes in notebooks/$(NOTEBOOK)"

track-all-notebooks: ## Enable tracking for all notebooks when you want to commit code changes
	@echo "📝 Enabling tracking for all notebooks..."
	git update-index --no-skip-worktree notebooks/*.ipynb
	@echo "✅ Now git will detect changes in all notebooks"

untrack-all-notebooks: ## Disable tracking for all notebooks (default state)
	@echo "🔄 Disabling tracking for all notebooks..."
	git update-index --skip-worktree notebooks/*.ipynb
	@echo "✅ Now git will ignore execution changes in all notebooks"

# Notebook management
notebook-status: ## Check which notebooks have changes
	@echo "📊 Notebook Status:"
	@for notebook in notebooks/*.ipynb; do \
		if git diff --quiet "$$notebook" 2>/dev/null; then \
			echo "  ✅ $$(basename $$notebook) - No changes"; \
		else \
			echo "  📝 $$(basename $$notebook) - Has changes"; \
		fi \
	done

track-notebook: ## Enable tracking for a specific notebook (usage: make track-notebook NOTEBOOK=filename.ipynb)
	@if [ -z "$(NOTEBOOK)" ]; then \
		echo "❌ Please specify NOTEBOOK=filename.ipynb"; \
		exit 1; \
	fi
	@echo "📝 Enabling tracking for notebooks/$(NOTEBOOK)..."
	git update-index --no-skip-worktree "notebooks/$(NOTEBOOK)"
	@echo "✅ Now git will detect changes in notebooks/$(NOTEBOOK)"

untrack-notebook: ## Disable tracking for a specific notebook (usage: make untrack-notebook NOTEBOOK=filename.ipynb)
	@if [ -z "$(NOTEBOOK)" ]; then \
		echo "❌ Please specify NOTEBOOK=filename.ipynb"; \
		exit 1; \
	fi
	@echo "🔄 Disabling tracking for notebooks/$(NOTEBOOK)..."
	git update-index --skip-worktree "notebooks/$(NOTEBOOK)"
	@echo "✅ Now git will ignore execution changes in notebooks/$(NOTEBOOK)"

track-all-notebooks: ## Enable tracking for all notebooks when you want to commit code changes
	@echo "📝 Enabling tracking for all notebooks..."
	git update-index --no-skip-worktree notebooks/*.ipynb
	@echo "✅ Now git will detect changes in all notebooks"

untrack-all-notebooks: ## Disable tracking for all notebooks (default state)
	@echo "🔄 Disabling tracking for all notebooks..."
	git update-index --skip-worktree notebooks/*.ipynb
	@echo "✅ Now git will ignore execution changes in all notebooks"

clean-notebook: ## Clean outputs from a specific notebook (usage: make clean-notebook NOTEBOOK=file.ipynb)
	@if [ -z "$(NOTEBOOK)" ]; then \
		echo "❌ Usage: make clean-notebook NOTEBOOK=file.ipynb"; \
		exit 1; \
	fi
	@echo "🧹 Cleaning notebooks/$(NOTEBOOK)..."
	docker-compose exec app python3 scripts/clean_notebook.py "notebooks/$(NOTEBOOK)"

clean-and-commit-notebook: ## Clean and commit a notebook (usage: make clean-and-commit-notebook NOTEBOOK=file.ipynb MSG="commit message")
	@if [ -z "$(NOTEBOOK)" ] || [ -z "$(MSG)" ]; then \
		echo "❌ Usage: make clean-and-commit-notebook NOTEBOOK=file.ipynb MSG=\"commit message\""; \
		exit 1; \
	fi
	@echo "🧹 Cleaning and committing notebooks/$(NOTEBOOK)..."
	git update-index --no-skip-worktree "notebooks/$(NOTEBOOK)"
	docker-compose exec app python3 scripts/clean_notebook.py "notebooks/$(NOTEBOOK)"
	git add "notebooks/$(NOTEBOOK)"
	git commit -m "$(MSG)"
	git update-index --skip-worktree "notebooks/$(NOTEBOOK)"
	@echo "✅ Notebook cleaned, committed, and tracking disabled"

clean-all-notebooks: ## Clean outputs from all notebooks
	@echo "🧹 Cleaning all notebooks..."
	@for notebook in notebooks/*.ipynb; do \
		echo "Cleaning $$(basename $$notebook)..."; \
		docker-compose exec app python3 scripts/clean_notebook.py "$$notebook"; \
	done
	@echo "✅ All notebooks cleaned"
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

# Jupyter and Notebooks
jupyter: ## Open Jupyter Lab in browser (after make up)
	@echo "📝 Opening Jupyter Lab..."
	@echo "🔗 URL: http://localhost:8888"
	@echo "📁 Notebooks available in: /notebooks/"
	@command -v open >/dev/null 2>&1 && open http://localhost:8888 || echo "Open http://localhost:8888 in your browser"

notebook-test: ## Test all notebooks using Docker environment
	@echo "🧪 Testing notebooks with Docker environment..."
	docker-compose exec app poetry run jupyter nbconvert --to notebook --execute --inplace notebooks/01_environment_test.ipynb
	docker-compose exec app poetry run jupyter nbconvert --to notebook --execute --inplace notebooks/02_math_tools_test.ipynb
	docker-compose exec app poetry run jupyter nbconvert --to notebook --execute --inplace notebooks/03_agent_test.ipynb
	docker-compose exec app poetry run jupyter nbconvert --to notebook --execute --inplace notebooks/04_database_test.ipynb
	@echo "✅ All notebooks tested successfully"

notebook-env: ## Test environment notebook specifically
	@echo "🧪 Testing environment notebook..."
	docker-compose exec app poetry run jupyter nbconvert --to notebook --execute --inplace notebooks/01_environment_test.ipynb
	@echo "✅ Environment test completed"

notebook-math: ## Test math tools notebook
	@echo "🧪 Testing math tools notebook..."
	docker-compose exec app poetry run jupyter nbconvert --to notebook --execute --inplace notebooks/02_math_tools_test.ipynb

notebook-agent: ## Test agent notebook
	@echo "🧪 Testing agent notebook..."
	docker-compose exec app poetry run jupyter nbconvert --to notebook --execute --inplace notebooks/03_agent_test.ipynb

notebook-db: ## Test database notebook
	@echo "🧪 Testing database notebook..."
	docker-compose exec app poetry run jupyter nbconvert --to notebook --execute --inplace notebooks/04_database_test.ipynb

# Development workflow shortcuts
setup: check-deps install ## Complete Docker-based setup
	@echo "🎉 Docker setup complete!"
	@echo ""
	@echo "🚀 Quick Start:"
	@echo "  make dev        - Start development environment"
	@echo "  make quick-start - Start services in background"
	@echo ""
	@echo "📝 Notebooks:"
	@echo "  Jupyter Lab will be available at http://localhost:8888"
	@echo "  All project dependencies are pre-installed in the container"
	@echo "  Use 'make notebook-test' to test all notebooks"

quick-start: up ## Quick start (assumes everything is built)
	@echo "🚀 Services started!"
	@echo "📱 Streamlit App: http://localhost:8501"
	@echo "� Jupyter Lab: http://localhost:8888"
	@echo "�🐘 Database: localhost:5432"
	@echo ""
	@echo "📋 Useful commands:"
	@echo "  make logs       - View all logs"
	@echo "  make jupyter    - Open Jupyter Lab"
	@echo "  make notebook-test - Test all notebooks"

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

# Notebook management
notebook-status: ## Check which notebooks have changes
	@echo "� Notebook Status:"
	@for notebook in notebooks/*.ipynb; do \
		if git diff --quiet "$$notebook" 2>/dev/null; then \
			echo "  ✅ $$(basename $$notebook) - No changes"; \
		else \
			echo "  📝 $$(basename $$notebook) - Has changes"; \
		fi \
	done

clean-notebooks-local: ## Clean notebook outputs locally (for development)
	@echo "🧹 Cleaning notebook outputs locally..."
	@for notebook in notebooks/*.ipynb; do \
		python3 scripts/clean_notebook.py < "$$notebook" > "$$notebook.tmp" && mv "$$notebook.tmp" "$$notebook"; \
	done
	@echo "✅ Local notebook outputs cleaned"
