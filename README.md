# Vanna AI 2.0 - Natural Language to SQL API

Production-ready FastAPI application that converts natural language questions to SQL queries using OpenAI, with persistent learning via ChromaDB.

## Features

- **Natural Language to SQL**: Convert questions to MySQL queries using GPT-4
- **Persistent Memory**: ChromaDB stores successful query patterns for learning
- **Context Isolation**: User and conversation-based memory sandboxing
- **Self-Learning**: System improves accuracy over time from successful queries
- **Plotly Charts**: Automatic chart generation for visualizable data
- **Health Monitoring**: Component-level health checks
- **PHP Compatible**: Simple JSON API for easy PHP integration

## Quick Start

### 1. Configure Environment

Copy and edit the `.env` file:

```bash
# Required: Set your OpenAI API key
OPENAI_API_KEY=sk-your-key-here
# or
OPENAI_API_PROJECT_KEY=sk-proj-your-key-here

# Optional: Change model (default: gpt-4)
OPENAI_MODEL=gpt-4
```

### 2. Start Services

```bash
docker-compose up -d
```

This repo includes a mock MySQL schema + seed data at `mysql/init/001_mock_schema_and_seed.sql`.

Note: MySQL init scripts only run the first time the database volume is created. To reseed/reset the mock data:

```bash
docker-compose down -v
docker-compose up -d
```

This starts:
- **MySQL** on port 3306
- **ChromaDB** on port 8001
- **Vanna API** on port 8000

### 3. Verify Health

```bash
curl http://localhost:8000/health
```

### 4. Test Natural Language → SQL (and Memory)

Ask the same question twice. The second call should typically return `from_cache: true`.

```bash
curl -s http://localhost:8000/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"question":"Top 5 products by revenue","user":"demo","conversation_id":"conv-1"}'

curl -s http://localhost:8000/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"question":"Top 5 products by revenue","user":"demo","conversation_id":"conv-1"}'
```

## API Endpoints

### POST /api/chat

Convert natural language to SQL and execute.

**Request:**
```json
{
  "question": "Show me all users created this month",
  "user": "user@example.com",
  "conversation_id": "conv-123"
}
```

**Response:**
```json
{
  "answer": "Found 15 results.",
  "sql_query": "SELECT * FROM users WHERE created_at >= DATE_FORMAT(NOW(), '%Y-%m-01')",
  "data": [{"id": 1, "name": "John", ...}],
  "chart": {"data": [...], "layout": {...}},
  "confidence": 0.9,
  "from_cache": false
}
```

### GET /health

Check system health.

**Response:**
```json
{
  "status": "healthy",
  "components": {
    "chromadb": {"status": "healthy", "collection_count": 42},
    "mysql": {"status": "healthy", "database": "vanna"},
    "openai": {"status": "healthy", "model": "gpt-4"}
  },
  "timestamp": "2024-01-15T10:30:00"
}
```

### GET /api/memory/{user}/{conversation_id}

Retrieve stored SQL patterns for a user/conversation.

**Response:**
```json
{
  "user": "user@example.com",
  "conversation_id": "conv-123",
  "memories": [
    {
      "id": "abc123",
      "question": "Show all users",
      "sql_query": "SELECT * FROM users",
      "timestamp": "2024-01-15T10:00:00",
      "confidence": 0.95
    }
  ],
  "total_count": 1
}
```

## PHP Integration

```php
<?php
class VannaClient {
    private string $baseUrl;
    
    public function __construct(string $baseUrl = 'http://localhost:8000') {
        $this->baseUrl = $baseUrl;
    }
    
    public function chat(string $question, string $user, string $conversationId): array {
        $ch = curl_init($this->baseUrl . '/api/chat');
        
        curl_setopt_array($ch, [
            CURLOPT_POST => true,
            CURLOPT_RETURNTRANSFER => true,
            CURLOPT_HTTPHEADER => ['Content-Type: application/json'],
            CURLOPT_POSTFIELDS => json_encode([
                'question' => $question,
                'user' => $user,
                'conversation_id' => $conversationId,
            ]),
        ]);
        
        $response = curl_exec($ch);
        $httpCode = curl_getinfo($ch, CURLINFO_HTTP_CODE);
        curl_close($ch);
        
        if ($httpCode !== 200) {
            throw new Exception("API error: $httpCode");
        }
        
        return json_decode($response, true);
    }
    
    public function health(): array {
        $response = file_get_contents($this->baseUrl . '/health');
        return json_decode($response, true);
    }
    
    public function getMemories(string $user, string $conversationId): array {
        $url = $this->baseUrl . "/api/memory/$user/$conversationId";
        $response = file_get_contents($url);
        return json_decode($response, true);
    }
}

// Usage
$vanna = new VannaClient('http://localhost:8000');

try {
    $result = $vanna->chat(
        'Show me top 10 customers by revenue',
        'admin@company.com',
        'session-' . session_id()
    );
    
    echo "SQL: " . $result['sql_query'] . "\n";
    echo "Answer: " . $result['answer'] . "\n";
    
    if (!empty($result['data'])) {
        foreach ($result['data'] as $row) {
            print_r($row);
        }
    }
    
    if (!empty($result['chart'])) {
        // Pass to Plotly.js on frontend
        echo json_encode($result['chart']);
    }
} catch (Exception $e) {
    echo "Error: " . $e->getMessage();
}
```

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | Yes* | - | OpenAI API key |
| `OPENAI_API_PROJECT_KEY` | Yes* | - | Alternative: OpenAI project key |
| `OPENAI_MODEL` | No | gpt-4 | OpenAI model to use |
| `OPENAI_EMBEDDING_MODEL` | No | text-embedding-3-small | OpenAI embedding model used for Chroma memory |
| `MYSQL_DO_HOST` | Yes | - | MySQL host |
| `MYSQL_DO_PORT` | No | 3306 | MySQL port |
| `MYSQL_DO_USER` | Yes | - | MySQL username |
| `MYSQL_DO_PASSWORD` | Yes | - | MySQL password |
| `MYSQL_DO_DATABASE` | Yes | - | MySQL database name |
| `CHROMA_HOST` | Yes | localhost | ChromaDB host |
| `CHROMA_PORT` | No | 8000 | ChromaDB port |
| `CHROMA_COLLECTION_NAME` | No | vanna_memories | ChromaDB collection |
| `APP_ENV` | No | production | Environment (development enables reload) |
| `LOG_LEVEL` | No | INFO | Logging level |

*One of `OPENAI_API_KEY` or `OPENAI_API_PROJECT_KEY` is required.

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   PHP Frontend  │────▶│   FastAPI App   │────▶│     MySQL       │
└─────────────────┘     └────────┬────────┘     └─────────────────┘
                                 │
                                 │
                        ┌────────▼────────┐
                        │    ChromaDB     │
                        │  (Vector Store) │
                        └─────────────────┘
                                 │
                        ┌────────▼────────┐
                        │     OpenAI      │
                        │   (GPT-4 LLM)   │
                        └─────────────────┘
```

## Memory & Learning System

1. **Question Received**: User asks natural language question
2. **Semantic Search**: ChromaDB searches for similar past questions
3. **Cache Hit**: If very similar question found (distance < 0.1), return cached SQL
4. **SQL Generation**: OpenAI generates SQL with schema context + similar examples
5. **Execution**: SQL runs against MySQL
6. **Learning**: Successful queries saved to ChromaDB with embeddings
7. **Improvement**: Future similar questions benefit from past successes

### Context Isolation

Memories are filtered by `user` + `conversation_id` metadata:
- Different users have separate memory sandboxes
- Different conversations maintain separate context
- Cross-conversation learning possible by querying user-level memories

## Development

```bash
# Run in development mode (auto-reload)
APP_ENV=development docker-compose up

# View logs
docker-compose logs -f app

# Rebuild after code changes
docker-compose up --build

# Stop all services
docker-compose down

# Stop and remove volumes (reset data)
docker-compose down -v
```

## API Documentation

Interactive API docs available at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## License

MIT
