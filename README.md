# 🤖 ReAct Agent for Integral Calculus

An intelligent agent usin## 🚀 Quick Start

### Prerequisites
- Docker and Docker Compose
- Google AI Studio API Key ([Get it here](https://aistudio.google.com/app/apikey))

### 1. Clone and Setup
```bash
git clone <repository-url>
cd react-integral-agent

# Verify project structure
./scripts/verify_setup.sh

# Copy environment template
cp .env.example .env
```

### 2. Configure Environment
Edit `.env` file with your configuration:
```bash
# Required: Add your Google API key
GOOGLE_API_KEY=your_actual_google_api_key_here

# Optional: Adjust other settings as needed
DEBUG=true
ENVIRONMENT=development
```

### 3. Launch with Docker
```bash
# Start all services (PostgreSQL + App)
make up

# Or manually:
docker-compose up --build
```

### 4. Verify Installation
```bash
# Run tests to ensure everything works
make test

# Check service health
make health

# View logs
make logs
```

### 5. Access Services
- **🌐 Streamlit App**: http://localhost:8501
- **📝 Jupyter Lab**: http://localhost:8888  
- **🗄️ Database**: localhost:5432 (user: agent_user, pass: agent_pass)

### 6. Test the Agent
Open Streamlit (http://localhost:8501) and try:
```
Calculate the integral of x^2 from 0 to 3 and visualize the area under the curve
``` AI** to solve mathematical integrals and visualize the area under the curve, with full persistence in PostgreSQL.

## 🚀 Features

- **🧠 ReAct Agent**: Intelligent reasoning and action with LangGraph
- **🔧 BigTool**: Automatic tool selection via semantic search
- **🤖 Gemini AI**: Google LLM for advanced mathematical reasoning
- **📊 Visualization**: Interactive plots with Matplotlib and Plotly
- **💾 Persistence**: PostgreSQL for conversations, cache, and metrics
- **🌐 Web Interface**: Streamlit with persistent state
- **🐳 Docker**: Full deployment with Docker Compose

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Streamlit UI  │────│  Persistent      │────│   PostgreSQL    │
│                 │    │  ReAct Agent     │    │   Database      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                       ┌────────┴────────┐
                       │    BigTool      │
                       │   Tool Registry │
                       └─────────────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        │                       │                       │
  ┌─────────────┐    ┌─────────────────┐    ┌─────────────────┐
  │ IntegralTool│    │   PlotTool      │    │  AnalysisTool   │
  │             │    │                 │    │                 │
  └─────────────┘    └─────────────────┘    └─────────────────┘
```

## 🛠️ Tech Stack

- **AI/Agents**: LangChain Core + LangGraph + BigTool
- **LLM**: Google Gemini (AI Studio)
- **Database**: PostgreSQL with vector extension
- **Frontend**: Streamlit
- **Math**: SciPy, NumPy, SymPy, Matplotlib
- **Containerization**: Docker + Docker Compose

## � Development Status

**Current Phase**: ✅ **Phase 1 Complete** - Architecture Base and Foundations

### ✅ Implemented Features:
- 🏗️ **Complete project structure** with modular architecture
- ⚙️ **Advanced configuration system** with environment validation
- 🔧 **Robust exception handling** with structured error reporting
- 📝 **Structured logging** with correlation IDs and JSON formatting
- 📊 **Data models** with Pydantic validation for conversations and agent state
- 💾 **Database layer** with sync/async connection management
- 🔒 **Mathematical expression validators** with security features
- 🌐 **Streamlit UI foundation** with chat interface and configuration
- 🧪 **Testing structure** ready for comprehensive test coverage

### 🔄 Next Phases:
- **Phase 2**: Mathematical Tools Implementation (IntegralTool, PlotTool, AnalysisTool)
- **Phase 3**: ReAct Agent with LangGraph integration
- **Phase 4**: Full Streamlit UI with visualizations
- **Phase 5**: Complete persistence and database features

## 📦 Quick Installation

### 🚀 For New Contributors (One-Command Setup)

```bash
# 1. Clone the repository
git clone <repository-url>
cd Agent_1

# 2. Complete Docker-based setup (handles everything!)
make setup
make quick-start

# 3. Access the applications
# 📱 Streamlit App: http://localhost:8501
# 📝 Jupyter Lab: http://localhost:8888
# 🐘 Database: localhost:5432
```

**That's it!** 🎉 No need to install Python, Poetry, or any dependencies locally. Docker handles everything.

### 📝 Using Notebooks

The project includes interactive Jupyter notebooks for testing and development:

```bash
# Open Jupyter Lab in browser
make jupyter

# Test all notebooks at once
make notebook-test

# Test specific notebooks
make notebook-env     # Environment tests
make notebook-math    # Mathematical tools
make notebook-agent   # ReAct agent tests
make notebook-db      # Database tests
```

All notebooks run with the **complete project environment** - no setup required!

### 🔧 Development Commands

```bash
# View all available commands
make help

# Monitor services
make status
make logs

# Quality checks
make test
make lint
make format

# Database operations
make db-shell
make db-reset
```

### 🐳 Why Docker-First?

- ✅ **Reproducible**: Same environment everywhere
- ✅ **Isolated**: Doesn't affect your local setup  
- ✅ **Complete**: Includes PostgreSQL, Jupyter, and all dependencies
- ✅ **Easy**: One command to start everything
- ✅ **Consistent**: Python 3.11 with exact package versions

## 💻 Alternative: Local Development

If you prefer local development without Docker:

### 1. Set up environment variables
Create `.env` file with your Gemini API key:
```bash
GOOGLE_API_KEY=your_google_api_key_here
DATABASE_URL=postgresql://agent_user:secure_password@localhost:5432/react_agent_db
```

### 2. Install dependencies
```bash
# Install Poetry if not installed
curl -sSL https://install.python-poetry.org | python3 -

# Install project dependencies
poetry install
```

### 3. Start PostgreSQL
```bash
# Database only
docker-compose up postgres -d
```

### 4. Run the application
```bash
# Activate virtual environment
poetry shell

# Run Streamlit
poetry run streamlit run app/main.py

# Or run Jupyter
poetry run jupyter lab notebooks/
```

## 🧪 Testing

```bash
# Unit tests
poetry run pytest tests/unit/ -v

# Integration tests
poetry run pytest tests/integration/ -v

# Coverage tests
poetry run pytest --cov=app --cov-report=html

# Specific persistence tests
poetry run pytest tests/unit/test_persistence.py -v

# BigTool tests
poetry run pytest tests/unit/test_bigtool_integration.py -v
```

## 📊 Usage Examples

### Simple Integral Calculation
```
👤 User: "Calculate the integral of x^2 from 0 to 2"

🤖 Agent: 
Think: I need to compute the definite integral of x^2 over [0,2]
Act: Using IntegralTool with function="x**2", lower=0, upper=2
Observe: Result = 8/3 ≈ 2.667

The integral of x² from 0 to 2 is 8/3 ≈ 2.667
```

### Visualization with Area
```
👤 User: "Show the plot of sin(x) and the area under the curve from 0 to π"

🤖 Agent:
Think: I need to compute the integral and create a visualization
Act: IntegralTool → PlotTool with shaded area
Observe: Integral = 2.0, plot generated

[Plot with shaded area showing sin(x) from 0 to π]
The area under the curve of sin(x) from 0 to π is 2.0
```

## 🔧 Advanced Configuration

### Additional Environment Variables
```bash
# BigTool configuration
BIGTOOL_INDEX_BATCH_SIZE=100
BIGTOOL_SEARCH_LIMIT=5

# Calculation cache
CALCULATION_CACHE_TTL_HOURS=24
CONVERSATION_HISTORY_LIMIT=100

# Logging
DEBUG=True
LOG_LEVEL=INFO
```

### PostgreSQL Extensions
```sql
-- Required extensions
CREATE EXTENSION IF NOT EXISTS vector;  -- For embeddings
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";  -- For UUIDs
```

## 📈 Monitoring

### Available Metrics
- **Tool usage**: Frequency and performance
- **Cache hit rate**: Calculation cache efficiency  
- **Response time**: Agent latency
- **Conversations**: Persistent history

### Accessing Metrics
```python
# In Streamlit sidebar
st.sidebar.subheader("📊 Statistics")
tool_stats = st.session_state.tool_tracker.get_tool_statistics()
st.sidebar.json(tool_stats)
```

## 🔄 Persistence

### Persistent Data
- ✅ **Conversations**: Full history across sessions
- ✅ **Calculation cache**: Avoids recalculating integrals
- ✅ **Usage metrics**: Tool usage analysis
- ✅ **Agent state**: Recovery after restarts

### Database
```sql
-- Main tables
conversations     -- Conversation sessions
messages         -- User/agent messages  
calculation_results  -- Integral cache
tool_usage       -- Tool usage logs
```

## 🚀 Production Deployment

### Docker Compose (Recommended)
```bash
# Production
docker-compose -f docker-compose.yml up -d

# With pgAdmin for development
docker-compose --profile dev up -d
```

### Production Variables
```bash
# Security
POSTGRES_PASSWORD=<strong-password>
DATABASE_URL=postgresql://user:pass@host:5432/db

# Scalability
STREAMLIT_SERVER_MAX_UPLOAD_SIZE=200
BIGTOOL_INDEX_BATCH_SIZE=1000
```

## 🤝 Contributing

1. Fork the project
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License. See `LICENSE` for details.


---
