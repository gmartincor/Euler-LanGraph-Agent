# ReAct Agent for Integral Calculus

A professional intelligent agent using **ReAct (Reasoning and Acting)** methodology to solve mathematical integrals and visualize results with full persistence in PostgreSQL.

## Overview

This project implements a sophisticated ReAct (Reasoning and Acting) agent that combines advanced AI capabilities with mathematical computation tools. The agent can understand natural language requests for integral calculus problems, reason about the appropriate solution approach, execute calculations, and generate visualizations.

## Features

- **ReAct Agent**: Intelligent reasoning and action cycles using LangGraph
- **BigTool Integration**: Automatic tool selection via semantic search
- **Google Gemini 1.5 Flash**: Advanced LLM for mathematical reasoning
- **Interactive Visualizations**: Dynamic plots with Matplotlib and Plotly
- **Full Persistence**: PostgreSQL for conversations, cache, and metrics
- **Web Interface**: Professional Streamlit application
- **Docker Deployment**: Complete containerized solution

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    PRESENTATION LAYER                           │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────┐  │
│  │  Streamlit UI   │    │  Jupyter Lab    │    │   REST API  │  │
│  └─────────┬───────┘    └─────────────────┘    └─────────────┘  │
└───────────┬┴─────────────────────────────────────────────────────┘
            │ User Input: "Calculate integral of x^2 from 0 to 3"
            ▼
┌─────────────────────────────────────────────────────────────────┐
│                  LANGGRAPH REACT WORKFLOW                       │
│                                                                 │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────────┐    │
│  │   THINK     │────▶│     ACT     │────▶│    OBSERVE      │    │
│  │             │     │             │     │                 │    │
│  │ • Analyze   │     │ • Select    │     │ • Evaluate      │    │
│  │   request   │     │   tools     │     │   results       │    │
│  │ • Plan      │     │ • Execute   │     │ • Check         │    │
│  │   approach  │     │   actions   │     │   completion    │    │
│  │ • Reason    │     │ • Call APIs │     │ • Learn from    │    │
│  │   about     │     │ • Calculate │     │   feedback      │    │
│  │   math      │     │   integrals │     │ • Decide next   │    │
│  └─────┬───────┘     └─────┬───────┘     └───────┬─────────┘    │
│        │                   │                     │              │
│        │                   ▼                     │              │
│        │    ┌─────────────────────────────────┐  │              │
│        │    │      Google Gemini 1.5 Flash   │  │              │
│        │    │   (Mathematical Reasoning &     │  │              │
│        │    │    Natural Language Processing) │  │              │
│        │    └─────────────────────────────────┘  │              │
│        │                                         │              │
│        └─────────────────────────────────────────┴──────────────┤
│                                                                 │
│        ┌──────────── WORKFLOW CYCLE ────────────────────────┐    │
│        │ Continue until problem is solved OR max iterations │    │
│        └─────────────────────────────────────────────────────┘   │
└─────────────────┬───────────────────────────────────────────────┘
                  │
┌─────────────────┴───────────────────────────────────────────────┐
│                     TOOL MANAGEMENT LAYER                      │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │               BigTool Registry                          │    │
│  │          (Semantic Tool Selection)                      │    │
│  │                                                         │    │
│  │  "I need to calculate an integral" → IntegralTool      │    │
│  │  "Show me a plot" → PlotTool                           │    │
│  │  "Analyze the result" → AnalysisTool                   │    │
│  └─────────────────┬───────────────────────────────────────┘    │
│                    │                                            │
│  ┌─────────────────▼───────────────────────────────────────┐    │
│  │  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐    │    │
│  │  │IntegralTool │   │  PlotTool   │   │AnalysisTool │    │    │
│  │  │             │   │             │   │             │    │    │
│  │  │• SymPy      │   │• Matplotlib │   │• Statistics │    │    │
│  │  │• SciPy      │   │• Plotly     │   │• Validation │    │    │
│  │  │• Numerical  │   │• Interactive│   │• Error Check│    │    │
│  │  │  Integration│   │  Plots      │   │• Result Eval│    │    │
│  │  └─────────────┘   └─────────────┘   └─────────────┘    │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────┬───────────────────────────────────────────────┘
                  │
┌─────────────────┴───────────────────────────────────────────────┐
│                     PERSISTENCE LAYER                          │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────┐  │
│  │   PostgreSQL    │    │   Vector Store  │    │   Cache     │  │
│  │                 │    │   (Embeddings)  │    │   (Memory)  │  │
│  │• Conversations  │    │• Tool Vectors   │    │• Results    │  │
│  │• Agent State    │    │• Semantic Index │    │• Sessions   │  │
│  │• Workflow Steps │    │• Search History │    │• Temp Data  │  │
│  │• Results Cache  │    │• User Patterns  │    │• Tool Cache │  │
│  └─────────────────┘    └─────────────────┘    └─────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

The system follows a layered architecture with clear separation of concerns:

- **Presentation Layer**: Streamlit web interface for user interaction
- **Agent Layer**: ReAct agent powered by LangGraph for reasoning and decision-making
- **Tool Layer**: Specialized mathematical tools managed by BigTool for intelligent selection
- **Data Layer**: PostgreSQL database for persistence and conversation history
- **LLM Integration**: Google Gemini 1.5 Flash for natural language processing and mathematical reasoning

The agent uses BigTool's semantic search to automatically select the most appropriate tools (IntegralTool, PlotTool, AnalysisTool) based on user requests, ensuring efficient and accurate responses.

## Technology Stack

- **AI Framework**: LangChain Core 0.3.72 + LangGraph 0.6.2
- **LLM Provider**: Google Gemini 1.5 Flash (AI Studio)
- **Tool Management**: BigTool 0.0.3 for intelligent tool selection
- **Database**: PostgreSQL with vector extensions
- **Frontend**: Streamlit for web interface
- **Mathematical Libraries**: SciPy, NumPy, SymPy, Matplotlib
- **Deployment**: Docker + Docker Compose
- **Development**: Poetry for dependency management

## Quick Start

### Prerequisites

- Docker and Docker Compose
- Google AI Studio API Key ([Get it here](https://aistudio.google.com/app/apikey))

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Agent_1
   ```

2. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env and add your Google API key
   ```

3. **Start services**
   ```bash
   make setup
   make up
   ```

4. **Verify installation**
   ```bash
   make test
   make health
   ```

### Access Points

- **Streamlit Application**: http://localhost:8501
- **Jupyter Lab**: http://localhost:8888
- **PostgreSQL**: localhost:5432 (user: agent_user, pass: agent_pass)

### Example Usage

Open the Streamlit interface and try:
```
Calculate the integral of x^2 from 0 to 3 and visualize the area under the curve
```


## Development

### Docker-Based Development (Recommended)

```bash
# View all available commands
make help

# Start development environment
make dev

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

### Local Development

If you prefer local development without Docker:

1. **Install Poetry**
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

2. **Install dependencies**
   ```bash
   poetry install
   ```

3. **Start PostgreSQL**
   ```bash
   docker-compose up postgres -d
   ```

4. **Run the application**
   ```bash
   poetry shell
   poetry run streamlit run app/main.py
   ```

### Jupyter Notebooks

Interactive notebooks for testing and development:

```bash
# Open Jupyter Lab
make jupyter

# Test all notebooks
make notebook-test

# Test specific notebooks
make notebook-env     # Environment tests
make notebook-math    # Mathematical tools
make notebook-agent   # ReAct agent tests
make notebook-db      # Database tests
```

## Testing

### Running Tests

```bash
# All tests
make test

# Unit tests only
poetry run pytest tests/unit/ -v

# Integration tests
poetry run pytest tests/integration/ -v

# Coverage report
poetry run pytest --cov=app --cov-report=html

# Specific test suites
poetry run pytest tests/unit/test_persistence.py -v
poetry run pytest tests/unit/test_bigtool_integration.py -v
```

## Configuration

### Environment Variables

Create a `.env` file with the following configuration:

```bash
# Required
GOOGLE_API_KEY=your_google_api_key_here
DATABASE_URL=postgresql://agent_user:agent_pass@localhost:5432/react_agent_db

# Optional
DEBUG=true
ENVIRONMENT=development
LOG_LEVEL=INFO

# BigTool Configuration
BIGTOOL_INDEX_BATCH_SIZE=100
BIGTOOL_SEARCH_LIMIT=5

# Cache Settings
CALCULATION_CACHE_TTL_HOURS=24
CONVERSATION_HISTORY_LIMIT=100
```

### Database Setup

PostgreSQL extensions required:
```sql
CREATE EXTENSION IF NOT EXISTS vector;      -- For embeddings
CREATE EXTENSION IF NOT EXISTS "uuid-ossp"; -- For UUIDs
```

## Usage Examples

### Simple Integral Calculation

```
User: "Calculate the integral of x^2 from 0 to 2"

Agent: 
Think: I need to compute the definite integral of x^2 over [0,2]
Act: Using IntegralTool with function="x**2", lower=0, upper=2
Observe: Result = 8/3 ≈ 2.667

The integral of x² from 0 to 2 is 8/3 ≈ 2.667
```

### Visualization with Area

```
User: "Show the plot of sin(x) and the area under the curve from 0 to π"

Agent:
Think: I need to compute the integral and create a visualization
Act: IntegralTool → PlotTool with shaded area
Observe: Integral = 2.0, plot generated

[Interactive plot with shaded area showing sin(x) from 0 to π]
The area under the curve of sin(x) from 0 to π is 2.0
```

## Persistence and Monitoring

### Persistent Data

The system maintains persistent state across sessions:

- **Conversations**: Full history across sessions
- **Calculation Cache**: Avoids recalculating integrals
- **Usage Metrics**: Tool usage analysis
- **Agent State**: Recovery after restarts

### Database Schema

```sql
-- Main tables
conversations        -- Conversation sessions
messages            -- User/agent messages  
calculation_results -- Integral cache
tool_usage          -- Tool usage logs
```

### Monitoring

Available metrics include:
- Tool usage frequency and performance
- Cache hit rate for calculations
- Agent response time
- Conversation history and analytics

Access metrics through the Streamlit sidebar:
```python
tool_stats = st.session_state.tool_tracker.get_tool_statistics()
st.sidebar.json(tool_stats)
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
