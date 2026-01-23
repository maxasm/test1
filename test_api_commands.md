# Vanna AI API Test Commands

This document provides curl commands to test the Vanna AI API endpoints. The server should be running on `localhost:8000` (default port).

## 1. Health Check
Check if the server is running.

```bash
curl -X GET http://localhost:8000/health
```

## 2. Synchronous Chat Endpoint
Send a natural language query and get SQL + results.

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Show me all tables",
    "conversation_id": "test-conv-1",
    "run_sql": true
  }'
```

## 3. Streaming Chat Endpoint (SSE)
Streaming chat endpoint for real-time responses.

```bash
curl -X POST http://localhost:8000/api/vanna/v2/chat_sse \
  -H "Content-Type: application/json" \
  -H "X-Conversation-Id: test-conv-2" \
  -d '{
    "message": "How many users are there?"
  }'
```

## 4. Seed Training Data
Seed the agent memory with DDL documentation and example queries.

```bash
curl -X POST http://localhost:8000/api/train/seed \
  -H "Content-Type: application/json" \
  -d '{}'
```

## 5. Check Training Status
Get the current training data status.

```bash
curl -X GET "http://localhost:8000/api/train/status"
```

## 6. Debug Environment
View environment configuration for debugging.

```bash
curl -X GET http://localhost:8000/debug/env
```

## 7. Debug Tools
View registered tools for debugging.

```bash
curl -X GET http://localhost:8000/debug/tools
```

## 8. Debug SQL
Test SQL execution directly (bypasses LLM).

```bash
curl -X GET http://localhost:8000/debug/sql
```

## 9. List Text Memories (Admin)
List all text memories stored in ChromaDB.

```bash
curl -X GET "http://localhost:8000/api/admin/memory/text?limit=10&offset=0"
```

## 10. Create Text Memory (Admin)
Add a new text memory.

```bash
curl -X POST http://localhost:8000/api/admin/memory/text \
  -H "Content-Type: application/json" \
  -d '{
    "content": "This is a test memory added via API"
  }'
```

## 11. List Golden Queries (Admin)
List all golden queries (question + SQL pairs).

```bash
curl -X GET "http://localhost:8000/api/admin/golden_queries?limit=10&offset=0"
```

## 12. Create Golden Query (Admin)
Add a new golden query.

```bash
curl -X POST http://localhost:8000/api/admin/golden_queries \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the top products by revenue?",
    "sql": "SELECT p.name, SUM(oi.line_total) as revenue FROM products p JOIN order_items oi ON p.id = oi.product_id GROUP BY p.id ORDER BY revenue DESC LIMIT 10"
  }'
```

## 13. Chat with Authentication Headers
If using signed header authentication (when VANNA_PROXY_HMAC_SECRET is set), include authentication headers:

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -H "X-User-Id: test-user-123" \
  -H "X-Conversation-Id: test-conv-3" \
  -d '{
    "message": "Show recent orders",
    "run_sql": true
  }'
```

## 14. Test with jq for Pretty Output
Use `jq` to format JSON responses:

```bash
curl -s http://localhost:8000/debug/env | jq .
curl -s http://localhost:8000/debug/tools | jq .
```

## 15. Batch Test Script
Create a test script to run multiple commands:

```bash
#!/bin/bash
BASE_URL="http://localhost:8000"

echo "1. Testing health endpoint..."
curl -s "$BASE_URL/health" | jq .

echo -e "\n2. Testing debug env..."
curl -s "$BASE_URL/debug/env" | jq .

echo -e "\n3. Testing debug tools..."
curl -s "$BASE_URL/debug/tools" | jq .

echo -e "\n4. Testing chat endpoint..."
curl -s -X POST "$BASE_URL/api/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "Show me all tables", "run_sql": false}' | jq .
```

## Notes

1. **Server Port**: Default is 8000. Change if using different port (set by VANNA_PORT environment variable).
2. **Authentication**: By default, uses PHP frontend authentication via headers/cookies. If `VANNA_PROXY_HMAC_SECRET` is set, uses signed header authentication.
3. **Database Connection**: Ensure MySQL database is accessible with credentials in `.env` file.
4. **ChromaDB**: Ensure ChromaDB is running if using persistent memory (default uses HTTP connection to `chroma:8000`).
5. **OpenAI API**: Ensure OpenAI API key is valid in `.env` file.

## Troubleshooting

If endpoints return errors:
- Check server logs: `python main.py` should show startup messages
- Verify dependencies: `pip install -r requirements.txt`
- Check environment variables: `.env` file should have required values
- Test database connection: MySQL should be accessible with SSL if configured
