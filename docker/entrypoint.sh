#!/bin/bash
set -e

# ReAct Agent Docker Entrypoint Script
# Professional startup script with health checks and error handling

echo "🤖 ReAct Agent - Starting Services..."
echo "=================================="

# Function to check if a service is running
check_service() {
    local service_name=$1
    local port=$2
    local max_attempts=30
    local attempt=1
    
    echo "🔍 Checking $service_name on port $port..."
    
    while [ $attempt -le $max_attempts ]; do
        if nc -z localhost $port 2>/dev/null; then
            echo "✅ $service_name is ready on port $port"
            return 0
        fi
        echo "⏳ Waiting for $service_name... (attempt $attempt/$max_attempts)"
        sleep 2
        attempt=$((attempt + 1))
    done
    
    echo "❌ $service_name failed to start after $max_attempts attempts"
    return 1
}

# Function to handle shutdown
cleanup() {
    echo ""
    echo "🛑 Shutting down services..."
    
    # Kill background processes
    if [ ! -z "$JUPYTER_PID" ]; then
        echo "📝 Stopping Jupyter Lab (PID: $JUPYTER_PID)..."
        kill -TERM $JUPYTER_PID 2>/dev/null || true
    fi
    
    if [ ! -z "$STREAMLIT_PID" ]; then
        echo "📊 Stopping Streamlit (PID: $STREAMLIT_PID)..."
        kill -TERM $STREAMLIT_PID 2>/dev/null || true
    fi
    
    # Wait for processes to terminate
    wait 2>/dev/null || true
    
    echo "✅ Services stopped gracefully"
    exit 0
}

# Set up signal handlers
trap cleanup SIGTERM SIGINT

# Environment validation
echo "🔧 Validating environment..."

# Check required environment variables
required_vars=("DATABASE_URL" "GOOGLE_API_KEY")
for var in "${required_vars[@]}"; do
    if [ -z "${!var}" ]; then
        echo "❌ Required environment variable $var is not set"
        exit 1
    fi
done

# Check if .env file exists, if not copy from example
if [ ! -f ".env" ] && [ -f ".env.example" ]; then
    echo "📋 Creating .env from .env.example..."
    cp .env.example .env
    echo "⚠️  Please update .env with your actual values"
fi

# Wait for database to be available
echo "🗄️  Waiting for database connection..."
if command -v nc >/dev/null 2>&1; then
    # Extract database host and port from DATABASE_URL
    DB_HOST=$(echo $DATABASE_URL | cut -d'@' -f2 | cut -d':' -f1)
    DB_PORT=$(echo $DATABASE_URL | cut -d':' -f4 | cut -d'/' -f1)
    
    if [ ! -z "$DB_HOST" ] && [ ! -z "$DB_PORT" ]; then
        max_attempts=30
        attempt=1
        while [ $attempt -le $max_attempts ]; do
            if nc -z $DB_HOST $DB_PORT 2>/dev/null; then
                echo "✅ Database is ready"
                break
            fi
            echo "⏳ Waiting for database... (attempt $attempt/$max_attempts)"
            sleep 2
            attempt=$((attempt + 1))
        done
        
        if [ $attempt -gt $max_attempts ]; then
            echo "❌ Database connection timeout"
            exit 1
        fi
    fi
fi

# Install dependencies if needed
echo "📦 Ensuring dependencies are installed..."
poetry install --no-root --quiet

# Run tests in development mode
if [ "${DEBUG:-false}" = "true" ] || [ "${ENVIRONMENT:-production}" = "development" ]; then
    echo "🧪 Running tests in development mode..."
    poetry run python -m pytest tests/ --tb=short -q || echo "⚠️  Some tests failed, but continuing..."
fi

echo "🚀 Starting services..."

# Start Jupyter Lab in background
echo "📝 Starting Jupyter Lab on port 8888..."
poetry run jupyter lab --config=/root/.jupyter/jupyter_lab_config.py > /var/log/jupyter.log 2>&1 &
JUPYTER_PID=$!
echo "📝 Jupyter Lab started with PID: $JUPYTER_PID"

# Start Streamlit in background
echo "📊 Starting Streamlit on port 8501..."
poetry run streamlit run app/main.py --server.port=8501 --server.address=0.0.0.0 > /var/log/streamlit.log 2>&1 &
STREAMLIT_PID=$!
echo "📊 Streamlit started with PID: $STREAMLIT_PID"

# Wait a bit for services to initialize
sleep 5

# Health checks
echo "🔍 Running health checks..."

# Check Jupyter Lab
if ! check_service "Jupyter Lab" 8888; then
    echo "❌ Jupyter Lab health check failed"
    cleanup
    exit 1
fi

# Check Streamlit  
if ! check_service "Streamlit" 8501; then
    echo "❌ Streamlit health check failed"
    cleanup
    exit 1
fi

echo ""
echo "✅ All services are running successfully!"
echo "=================================="
echo "🌐 Service URLs:"
echo "  📊 Streamlit App:    http://localhost:8501"
echo "  📝 Jupyter Lab:      http://localhost:8888"
echo "  🗄️  Database:         $DATABASE_URL"
echo ""
echo "📋 Available commands:"
echo "  make test            # Run tests"
echo "  make lint            # Run linting"
echo "  make logs            # View logs"
echo ""
echo "🛑 Press Ctrl+C to stop all services"
echo "=================================="

# Keep container running and wait for both processes
wait
