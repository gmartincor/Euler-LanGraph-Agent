#!/bin/bash

set -e

check_env() {
    if [ ! -f .env ]; then
        echo "GOOGLE_API_KEY=your_api_key_here" > .env
        echo "Edit .env with your API key"
        exit 1
    fi
}

case "$1" in
    up) check_env; docker-compose up --build ;;
    down) docker-compose down ;;
    clean) docker-compose down -v; docker system prune -f ;;
    logs) docker-compose logs -f ;;
    db) docker-compose exec postgres psql -U agent_user -d react_agent_db ;;
    shell) docker-compose exec app /bin/bash ;;
    jupyter) docker-compose exec app poetry run jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root ;;
    *) echo "Usage: $0 {up|down|clean|logs|db|shell|jupyter}" ;;
esac
