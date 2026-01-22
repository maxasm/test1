# Vanna AI 2.0 - Natural Language to SQL

Production-ready natural language to SQL application using Vanna AI 2.0 agent framework with OpenAI LLM, MySQL database, ChromaDB persistent memory, and a web dashboard.

## Overview

This application allows users to query a MySQL database using natural language. It includes:

- **Vanna AI 2.0**: Agent framework for natural language to SQL
- **OpenAI GPT-4**: Large language model for understanding queries
- **MySQL**: Database for storing and querying data
- **ChromaDB**: Vector database for persistent memory and learning
- **Web Dashboard**: Protected admin interface with passkey authentication

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   PHP Frontend  │────▶│  FastAPI Server │────▶│     MySQL       │
│   (External)    │     │   (Vanna 2.0)   │     │   Database      │
└─────────────────┘     └────────┬────────┘     └─────────────────┘
                                 │
                        ┌────────┴────────┐
                        ▼                 ▼
               ┌─────────────────┐ ┌─────────────────┐
               │    ChromaDB     │ │    Dashboard    │
               │ (Agent Memory)  │ │   (Port 8080)   │
               └─────────────────┘ └─────────────────┘
```

## Services & Ports

| Service | Port | Description |
|---------|------|-------------|
| MySQL | 3306 | Database server |
| ChromaDB | 8001 | Vector memory store |
| Vanna App | 8000 | Main FastAPI application |
| Dashboard | 8080 | Web admin interface |

## Quick Start

### Prerequisites

- Docker and Docker Compose
- OpenAI API key

### 1. Configure Environment

Edit the `.env` file with your settings:

```env
# Required: OpenAI API Key
OPENAI_API_KEY=your-openai-api-key-here
# Or use project key:
OPENAI_API_PROJECT_KEY=sk-proj-...

# Dashboard passkey (change this!)
DASHBOARD_PASSKEY=your-secure-passkey
```

### 2. Start All Services

```bash
docker-compose up -d --build
```

This starts all services:
- MySQL database on port **3306**
- ChromaDB on port **8001**
- Vanna FastAPI server on port **8000**
- Dashboard on port **8080**

### 3. Access the Dashboard

Open your browser and navigate to:

```
http://localhost:8080
```

Enter the passkey you configured in `.env` (`DASHBOARD_PASSKEY`) to access the dashboard.

### 4. Verify Services

Check the health of the main application:

```bash
curl http://localhost:8000/health
```

### 5. Seed Training Data (Optional)

To improve query accuracy, seed the agent with schema documentation:

```bash
curl -X POST http://localhost:8000/api/train/seed
```

## Dashboard

The web dashboard provides:

- **Service Status**: View status of all running services
- **Quick Actions**: Links to Vanna Chat, API docs, and health checks
- **System Info**: Current configuration and environment details

### Dashboard Authentication

The dashboard is protected by a passkey. Configure it in `.env`:

```env
DASHBOARD_PASSKEY=your-secure-passkey
DASHBOARD_PORT=8080
```

## API Endpoints

### Chat (Streaming)

```bash
POST /api/vanna/v2/chat_sse
Content-Type: application/json

{
  "message": "Show me all customers from California"
}
```

Headers for user identification:
- `X-User-Id`: User identifier
- `X-Conversation-Id`: Conversation identifier

### Health Check

```bash
GET /health
```

### Training

```bash
GET /api/train/status      # Get training status
POST /api/train/seed       # Seed training data
```

### Conversation History

```bash
GET /api/conversation/messages?limit=50
GET /api/conversation/events?limit=200
DELETE /api/conversation
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key | Required |
| `OPENAI_MODEL` | OpenAI model | `gpt-4` |
| `OPENAI_EMBEDDING_MODEL` | Embedding model | `text-embedding-3-small` |
| `MYSQL_DO_HOST` | MySQL host | `mysql` |
| `MYSQL_DO_PORT` | MySQL port | `3306` |
| `MYSQL_DO_DATABASE` | MySQL database | `vanna` |
| `MYSQL_DO_USER` | MySQL user | `vanna` |
| `MYSQL_DO_PASSWORD` | MySQL password | `vanna_password` |
| `CHROMA_HOST` | ChromaDB host | `chroma` |
| `CHROMA_PORT` | ChromaDB port | `8000` |
| `VANNA_HOST` | Server bind host | `0.0.0.0` |
| `VANNA_PORT` | Server port | `8000` |
| `DASHBOARD_PASSKEY` | Dashboard access passkey | `changeme` |
| `DASHBOARD_PORT` | Dashboard port | `8080` |
| `APP_ENV` | Environment | `production` |
| `LOG_LEVEL` | Logging level | `INFO` |

## Docker Commands

```bash
# Start all services
docker-compose up -d --build

# View logs
docker-compose logs -f

# View specific service logs
docker-compose logs -f dashboard
docker-compose logs -f app

# Stop all services
docker-compose down

# Stop and remove volumes (reset data)
docker-compose down -v

# Rebuild a specific service
docker-compose up -d --build dashboard
```

## Troubleshooting

### Dashboard Not Loading

1. Check if the container is running:
   ```bash
   docker-compose ps
   ```

2. View dashboard logs:
   ```bash
   docker-compose logs dashboard
   ```

### MySQL Connection Issues

Ensure MySQL is healthy:
```bash
docker-compose logs mysql
```

### ChromaDB Connection Issues

Check ChromaDB is running:
```bash
curl http://localhost:8001/api/v1/heartbeat
```

### OpenAI API Errors

Verify your API key is set in `.env`:
```bash
grep OPENAI .env
```

## File Structure

```
.
├── .env                 # Environment configuration
├── docker-compose.yml   # Docker services definition
├── Dockerfile           # Container build instructions
├── main.py              # Vanna FastAPI application
├── dashboard.py         # Web dashboard application
├── training_data.py     # Schema and training content
├── requirements.txt     # Python dependencies
└── mysql/
    └── init/            # MySQL initialization scripts
```
