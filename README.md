# Vanna AI 2.0 - Natural Language to SQL API

Production-ready FastAPI application using the **official Vanna AI 2.0 agent framework** for natural language to SQL conversion with OpenAI LLM, MySQL database, and ChromaDB persistent memory.

**Following official Vanna 2.0 docs:** https://vanna.ai/docs

## Features

- **Official Vanna AI 2.0 Framework**: Uses `Agent`, `RunSqlTool`, `VisualizeDataTool`, and memory tools
- **OpenAI LLM Integration**: `OpenAILlmService` for natural language understanding
- **MySQL Database**: `MySQLRunner` for SQL query execution
- **ChromaDB Agent Memory**: `ChromaAgentMemory` for persistent learning from successful queries
- **User-Aware Permissions**: Custom `UserResolver` for PHP frontend integration
- **Tool Memory**: Saves successful tool usage patterns for future reference
- **Streaming Chat**: SSE-based streaming responses with rich UI components
- **Built-in Web UI**: Pre-built `<vanna-chat>` component support
- **Health Monitoring**: Component-level health checks

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

### 4. Test with Web UI

Open http://localhost:8000 in your browser to access the built-in Vanna web interface.

### 5. Test via API (Streaming SSE)

```bash
# Using curl with SSE streaming
curl -N http://localhost:8000/api/vanna/v2/chat_sse \
  -H 'Content-Type: application/json' \
  -H 'X-User-Id: demo@example.com' \
  -H 'X-Conversation-Id: conv-1' \
  -d '{"message":"Top 5 products by revenue"}'
```

## API Endpoints

### POST /api/vanna/v2/chat_sse (Streaming)

Main chat endpoint using Server-Sent Events for streaming responses.

**Headers:**
- `X-User-Id`: User identifier (or use `vanna_user` cookie)
- `X-Conversation-Id`: Conversation ID (or use `vanna_conversation_id` cookie)

**Request:**
```json
{
  "message": "Show me all users created this month"
}
```

**Response (SSE stream):**
The response streams UI components including:
- Text responses
- SQL queries
- Data tables (`DataFrameComponent`)
- Charts (`ChartComponent`)
- Tool execution status

### GET /health

Check system health.

### GET /docs

OpenAPI/Swagger documentation.

### GET /

Built-in Vanna web UI.

## PHP Integration

```php
<?php
class VannaClient {
    private string $baseUrl;
    
    public function __construct(string $baseUrl = 'http://localhost:8000') {
        $this->baseUrl = $baseUrl;
    }
    
    /**
     * Send a chat message and receive streaming response.
     * For PHP, you may want to use a non-streaming approach or process SSE.
     */
    public function chat(string $message, string $user, string $conversationId): Generator {
        $ch = curl_init($this->baseUrl . '/api/vanna/v2/chat_sse');
        
        curl_setopt_array($ch, [
            CURLOPT_POST => true,
            CURLOPT_RETURNTRANSFER => false,
            CURLOPT_HTTPHEADER => [
                'Content-Type: application/json',
                'X-User-Id: ' . $user,
                'X-Conversation-Id: ' . $conversationId,
            ],
            CURLOPT_POSTFIELDS => json_encode(['message' => $message]),
            CURLOPT_WRITEFUNCTION => function($ch, $data) use (&$buffer) {
                // Process SSE data
                $buffer .= $data;
                return strlen($data);
            },
        ]);
        
        curl_exec($ch);
        curl_close($ch);
        
        // Parse SSE events from buffer
        return $this->parseSSE($buffer);
    }
    
    public function health(): array {
        $response = file_get_contents($this->baseUrl . '/health');
        return json_decode($response, true);
    }
    
    private function parseSSE(string $buffer): array {
        $events = [];
        $lines = explode("\n", $buffer);
        $currentData = '';
        
        foreach ($lines as $line) {
            if (strpos($line, 'data: ') === 0) {
                $currentData = substr($line, 6);
                if ($currentData && $currentData !== '[DONE]') {
                    $events[] = json_decode($currentData, true);
                }
            }
        }
        
        return $events;
    }
}

// Usage with session-based user context
session_start();

$vanna = new VannaClient('http://localhost:8000');

try {
    $events = $vanna->chat(
        'Show me top 10 customers by revenue',
        $_SESSION['user_email'] ?? 'guest@example.com',
        'session-' . session_id()
    );
    
    foreach ($events as $event) {
        // Handle different component types
        if (isset($event['type'])) {
            switch ($event['type']) {
                case 'text':
                    echo $event['content'];
                    break;
                case 'dataframe':
                    // Render table
                    break;
                case 'chart':
                    // Pass to Plotly.js
                    break;
            }
        }
    }
} catch (Exception $e) {
    echo "Error: " . $e->getMessage();
}
```

## Web Component Integration

Drop the Vanna chat component into any webpage:

```html
<!-- In your PHP/HTML template -->
<script src="https://img.vanna.ai/vanna-components.js"></script>
<vanna-chat 
    sse-endpoint="http://localhost:8000/api/vanna/v2/chat_sse"
    theme="light">
</vanna-chat>

<script>
// Set user context via cookies (picked up by UserResolver)
document.cookie = `vanna_user=${encodeURIComponent(userEmail)}; path=/`;
document.cookie = `vanna_conversation_id=${encodeURIComponent(sessionId)}; path=/`;
</script>
```

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | Yes* | - | OpenAI API key |
| `OPENAI_API_PROJECT_KEY` | Yes* | - | Alternative: OpenAI project key |
| `OPENAI_MODEL` | No | gpt-4 | OpenAI model to use |
| `MYSQL_DO_HOST` | Yes | - | MySQL host |
| `MYSQL_DO_PORT` | No | 3306 | MySQL port |
| `MYSQL_DO_USER` | Yes | - | MySQL username |
| `MYSQL_DO_PASSWORD` | Yes | - | MySQL password |
| `MYSQL_DO_DATABASE` | Yes | - | MySQL database name |
| `CHROMA_HOST` | No | localhost | ChromaDB host |
| `CHROMA_PORT` | No | 8000 | ChromaDB port |
| `CHROMA_COLLECTION_NAME` | No | vanna_memories | ChromaDB collection |
| `VANNA_HOST` | No | 0.0.0.0 | Server bind host |
| `VANNA_PORT` | No | 8000 | Server port |
| `LOG_LEVEL` | No | INFO | Logging level |

*One of `OPENAI_API_KEY` or `OPENAI_API_PROJECT_KEY` is required.

## Architecture (Vanna AI 2.0)

```
┌─────────────────┐     ┌─────────────────────────────────────────┐
│   PHP Frontend  │────▶│          Vanna AI 2.0 Agent             │
│  <vanna-chat>   │     │                                         │
└─────────────────┘     │  ┌─────────────┐  ┌──────────────────┐  │
                        │  │ UserResolver│  │   ToolRegistry   │  │
                        │  └─────────────┘  │  ┌────────────┐  │  │
                        │                   │  │ RunSqlTool │  │  │
                        │  ┌─────────────┐  │  ├────────────┤  │  │
                        │  │OpenAILlm    │  │  │VisualizeData│ │  │
                        │  │Service      │  │  ├────────────┤  │  │
                        │  └─────────────┘  │  │MemoryTools │  │  │
                        │                   │  └────────────┘  │  │
                        │  ┌─────────────┐  └──────────────────┘  │
                        │  │ChromaAgent  │                        │
                        │  │Memory       │                        │
                        │  └─────────────┘                        │
                        └───────────┬─────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
            ┌───────────┐   ┌───────────┐   ┌───────────┐
            │  MySQL    │   │ ChromaDB  │   │  OpenAI   │
            │ Database  │   │  Memory   │   │   API     │
            └───────────┘   └───────────┘   └───────────┘
```

## Vanna Tools

The agent has access to these tools:

| Tool | Description | Access Groups |
|------|-------------|---------------|
| `RunSqlTool` | Executes SQL queries against MySQL | admin, user |
| `VisualizeDataTool` | Generates charts from query results | admin, user |
| `SaveQuestionToolArgsTool` | Saves successful question→tool patterns | admin |
| `SearchSavedCorrectToolUsesTool` | Searches past successful patterns | admin, user |
| `SaveTextMemoryTool` | Saves arbitrary text memories | admin, user |

## Memory & Learning System

Vanna 2.0 uses `ChromaAgentMemory` to store:
1. **Successful tool usage patterns**: Question → Tool → Arguments → Result
2. **Text memories**: Golden queries, business rules, notes

When a user asks a question:
1. Agent searches memory for similar past questions
2. Retrieves successful SQL patterns as context
3. LLM generates new SQL informed by past successes
4. On success, the pattern is saved for future reference

### User Context Isolation

The `PHPFrontendUserResolver` extracts user identity from:
- Headers: `X-User-Id`, `X-Conversation-Id`
- Cookies: `vanna_user`, `vanna_conversation_id`

Memory is scoped per user, ensuring isolation between different users.

## Development

```bash
# Run in development mode
docker-compose up

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
