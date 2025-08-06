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
.PHONY: notebook-test notebook-clean notebook-status test-notebooks test-notebooks-full
.PHONY: test-environment test-agent-core test-workflow test-performance debug-system notebook-report
.PHONY: shell-app shell-db poetry-add poetry-remove workflow-test-quick workflow-test-full
.PHONY: jupyter jupyter-logs jupyter-install jupyter-shell verify vscode-setup
.PHONY: jupyter-kernel-setup jupyter-kernel-list jupyter-kernel-remove

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

# Jupyter Lab operations (professional development workflow)
jupyter: ## Open Jupyter Lab directly (alternative to make dev)
	@echo "📝 Starting Jupyter Lab only..."
	@$(DOCKER_COMPOSE) up -d postgres
	@sleep 3
	@$(DOCKER_COMPOSE) run --rm -p 8888:8888 $(APP_CONTAINER) poetry run jupyter lab --config=/root/.jupyter/jupyter_lab_config.py --no-browser --allow-root
	@echo "📝 Jupyter Lab available at: http://localhost:8888"

jupyter-logs: ## Show Jupyter Lab logs
	@echo "📝 Jupyter Lab logs:"
	@$(DOCKER_COMPOSE) exec $(APP_CONTAINER) tail -f /var/log/jupyter.log 2>/dev/null || echo "📝 Jupyter is running in console mode"

jupyter-install: ## Install package in Jupyter kernel (usage: make jupyter-install PKG=package_name)
	@if [ -z "$(PKG)" ]; then \
		echo "❌ Usage: make jupyter-install PKG=package_name"; \
		exit 1; \
	fi
	@echo "📦 Installing $(PKG) in Jupyter kernel..."
	@$(DOCKER_COMPOSE) exec $(APP_CONTAINER) poetry add $(PKG)
	@$(DOCKER_COMPOSE) restart $(APP_CONTAINER)
	@echo "✅ $(PKG) installed and kernel restarted"

jupyter-shell: ## Access shell inside the container
	@echo "🐚 Opening shell in container..."
	@$(DOCKER_COMPOSE) exec $(APP_CONTAINER) /bin/bash

# Jupyter Kernel for VS Code (professional development workflow)
jupyter-kernel-setup: ## Configure Jupyter kernel for VS Code using Docker
	@echo "⚙️  Configurando kernel Jupyter Docker para VS Code..."
	@chmod +x scripts/setup_kernel.sh
	@./scripts/setup_kernel.sh

jupyter-kernel-list: ## List available Jupyter kernels
	@echo "📋 Kernels disponibles:"
	@python scripts/setup_jupyter_kernel.py --list

jupyter-kernel-remove: ## Remove ReAct Agent Docker kernel
	@echo "🗑️  Removiendo kernel Docker..."
	@jupyter kernelspec remove react-agent-docker -f 2>/dev/null || echo "❌ Kernel no encontrado"
	@echo "✅ Kernel removido"

verify: ## Verify development environment setup
	@echo "🔍 Verifying development environment..."
	@$(DOCKER_COMPOSE) exec $(APP_CONTAINER) /app/scripts/verify_environment.sh

vscode-setup: ## Setup VS Code to use Docker kernel
	@echo "🔧 Setting up VS Code with Docker kernel..."
	@./scripts/setup_vscode_docker.sh

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
	@echo "🧹 Cleaning notebooks..."
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
			echo "  📝 $$(basename $$nb) - Modified"; \
		fi \
	done

# Professional Testing Suite (comprehensive validation)
test-notebooks: ## Run master test suite for comprehensive validation
	@echo "🎯 Running Master Test Suite..."
	@$(DOCKER_COMPOSE) up -d postgres
	@sleep 3
	@$(DOCKER_COMPOSE) exec $(APP_CONTAINER) python -c "import asyncio; from notebooks.run_master_tests import run_critical_tests; asyncio.run(run_critical_tests())"
	@echo "✅ Master test suite completed"

test-notebooks-full: ## Run complete test suite including performance tests
	@echo "🔄 Running Full Test Suite (Warning: ~15 minutes)..."
	@$(DOCKER_COMPOSE) up -d postgres
	@sleep 3
	@$(DOCKER_COMPOSE) exec $(APP_CONTAINER) python -c "import asyncio; from notebooks.run_master_tests import run_full_test_suite; asyncio.run(run_full_test_suite())"
	@echo "✅ Full test suite completed"

test-environment: ## Quick environment validation
	@echo "🔍 Running environment tests..."
	@$(DOCKER_COMPOSE) up -d postgres
	@sleep 2
	@$(DOCKER_COMPOSE) exec $(APP_CONTAINER) python -c "from notebooks.test_environment import run_env_tests; run_env_tests()"

test-agent-core: ## Test core agent components
	@echo "🧪 Testing agent core components..."
	@$(DOCKER_COMPOSE) up -d postgres
	@sleep 2
	@$(DOCKER_COMPOSE) exec $(APP_CONTAINER) python -c "import asyncio; from notebooks.test_agent_core import run_core_tests; asyncio.run(run_core_tests())"

test-workflow: ## Test workflow integration
	@echo "🔄 Testing workflow integration..."
	@$(DOCKER_COMPOSE) up -d postgres
	@sleep 2
	@$(DOCKER_COMPOSE) exec $(APP_CONTAINER) python -c "import asyncio; from notebooks.test_workflow import run_workflow_tests; asyncio.run(run_workflow_tests())"

test-performance: ## Run performance and load tests
	@echo "📈 Running performance tests..."
	@$(DOCKER_COMPOSE) up -d postgres
	@sleep 2
	@$(DOCKER_COMPOSE) exec $(APP_CONTAINER) python -c "import asyncio; from notebooks.test_performance import run_performance_tests; asyncio.run(run_performance_tests())"

debug-system: ## Run comprehensive system debugging
	@echo "🐛 Running system debugging..."
	@$(DOCKER_COMPOSE) up -d postgres
	@sleep 2
	@$(DOCKER_COMPOSE) exec $(APP_CONTAINER) python -c "import asyncio; from notebooks.debug_system import run_debug_analysis; asyncio.run(run_debug_analysis())"

notebook-report: ## Generate test report from notebooks
	@echo "📊 Generating notebook test report..."
	@$(DOCKER_COMPOSE) exec $(APP_CONTAINER) python scripts/generate_notebook_report.py
	@echo "✅ Report generated in reports/"

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
	@echo "Git Branch: $$(git branch --show-current 2>/dev/null || echo 'Not a git repo')"
	@echo ""
	@echo "🎯 Application URLs:"
	@echo "  📱 Streamlit (New UI): http://localhost:8501"
	@echo "  📝 Jupyter Lab: http://localhost:8888"  
	@echo "  🐘 PostgreSQL: localhost:5432"
	@echo ""
	@echo "🏗️ Phase 4 Features:"
	@echo "  ✅ Professional UI Components"
	@echo "  ✅ Modular Architecture (DRY/KISS/YAGNI)"
	@echo "  ✅ Real-time Chat Interface"
	@echo "  ✅ Metrics Dashboard"
	@echo "  ✅ Professional Styling"

ui-test: ## Test the new UI components
	@echo "🧪 Testing new UI components..."
	@$(POETRY_CMD) pytest tests/ui/ -v

ui-check: ## Check UI component health
	@echo "🔍 Checking UI component health..."
	@echo "📊 Checking state management..."
	@$(POETRY_CMD) python -c "from app.ui.state import get_state_manager; print('✅ State management OK')"
	@echo "🎨 Checking styling utilities..."
	@$(POETRY_CMD) python -c "from app.ui.utils import StyleManager; print('✅ Styling utilities OK')"
	@echo "💬 Checking chat components..."
	@$(POETRY_CMD) python -c "from app.ui.components import ChatComponent; print('✅ Chat components OK')"

ui-dev: up ## Start development with UI focus
	@echo "🎨 Starting UI-focused development..."
	@echo ""
	@echo "🚀 New Professional UI Features:"
	@echo "  📱 Modular chat interface with real-time updates"
	@echo "  📊 Live metrics dashboard with tool usage stats"
	@echo "  ⚙️  Professional configuration panel"
	@echo "  🎯 Type-safe state management"
	@echo "  🛡️  Graceful error handling with recovery"
	@echo ""
	@echo "🔗 Access Points:"
	@echo "  📱 Main Application: http://localhost:8501"
	@echo "  📝 Development Notebooks: http://localhost:8888"
	@echo "  🐘 Database: localhost:5432"
	@echo ""
	@echo "💡 UI Development Tips:"
	@echo "  • Edit files in app/ui/ for hot reload"
	@echo "  • Use 'make ui-check' to verify components"
	@echo "  • Check 'make logs-app' for Streamlit logs"
	@echo "Docker Compose: $$(docker-compose --version)"
	@echo "Directory: $$(pwd)"
	@echo "Git branch: $$(git branch --show-current 2>/dev/null || echo 'Not a git repo')"

# Professional workflow testing commands (DRY principle)
workflow-test-quick: ## Quick workflow test using consolidated script
	@echo "🧪 Running quick workflow test..."
	@$(DOCKER_COMPOSE) run --rm $(APP_CONTAINER) python scripts/test_workflow.py

workflow-test-full: ## Full workflow test with detailed diagnostics
	@echo "🧪 Running comprehensive workflow test..."
	@$(DOCKER_COMPOSE) run --rm $(APP_CONTAINER) python -m pytest tests/integration/test_unified_workflow.py -v

# Convenience aliases
install: build ## Alias for build
quick-start: dev ## Alias for dev
