"""
Vanna AI 2.0 FastAPI Application
Production-ready natural language to SQL using Vanna AI 2.0 agent framework
with OpenAI LLM, MySQL database, and ChromaDB persistent memory.

Following official Vanna 2.0 docs: https://vanna.ai/docs
"""

import os
import logging
import json
from datetime import datetime
from typing import Optional

from dotenv import load_dotenv
import structlog

load_dotenv()

# -----------------------------------------------------------------------------
# Logging Configuration
# -----------------------------------------------------------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO))
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)
logger = structlog.get_logger()

# -----------------------------------------------------------------------------
# Vanna AI 2.0 Imports (official framework)
# -----------------------------------------------------------------------------
from vanna import Agent
from vanna.core.registry import ToolRegistry
from vanna.core.user import UserResolver, User, RequestContext
from vanna.tools import RunSqlTool, VisualizeDataTool
from vanna.tools.agent_memory import (
    SaveQuestionToolArgsTool,
    SearchSavedCorrectToolUsesTool,
    SaveTextMemoryTool,
)
from vanna.servers.fastapi import VannaFastAPIServer
from vanna.integrations.openai import OpenAILlmService
from vanna.integrations.mysql import MySQLRunner
from vanna.integrations.chromadb import ChromaAgentMemory
import chromadb
from chromadb.config import Settings
import pymysql

from fastapi import Request
from starlette.responses import StreamingResponse

from training_data import get_all_training_content


# -----------------------------------------------------------------------------
# Custom HTTP-based ChromaDB Agent Memory for Docker
# -----------------------------------------------------------------------------
class HttpChromaAgentMemory(ChromaAgentMemory):
    """
    ChromaAgentMemory subclass that uses HttpClient for remote ChromaDB server.
    
    This allows the agent memory to connect to a ChromaDB server running in
    a separate Docker container instead of using local file persistence.
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 8000,
        collection_name: str = "tool_memories",
        embedding_function=None,
    ):
        # Don't call super().__init__() as it sets up PersistentClient
        # Instead, manually set attributes
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self._client = None
        self._collection = None
        self._embedding_function = embedding_function
        
        # Import ThreadPoolExecutor for async operations
        from concurrent.futures import ThreadPoolExecutor
        self._executor = ThreadPoolExecutor(max_workers=2)
    
    def _get_client(self):
        """Get or create ChromaDB HTTP client for remote server."""
        if self._client is None:
            self._client = chromadb.HttpClient(
                host=self.host,
                port=self.port,
                settings=Settings(anonymized_telemetry=False),
            )
        return self._client


# -----------------------------------------------------------------------------
# Custom User Resolver for PHP Frontend Integration
# -----------------------------------------------------------------------------
class PHPFrontendUserResolver(UserResolver):
    """
    User resolver that extracts user identity from request headers/body.
    
    PHP frontend sends user and conversation_id via:
    - Headers: X-User-Id, X-Conversation-Id
    - Or cookies: vanna_user, vanna_conversation_id
    
    This follows Vanna's placeholder/auth approach:
    https://vanna.ai/docs/placeholder/auth
    """

    async def resolve_user(self, request_context: RequestContext) -> User:
        user_id = (
            request_context.get_header("X-User-Id")
            or request_context.get_cookie("vanna_user")
            or "anonymous"
        )
        
        conversation_id = (
            request_context.get_header("X-Conversation-Id")
            or request_context.get_cookie("vanna_conversation_id")
            or "default"
        )
        
        group = "admin" if user_id.endswith("@admin") else "user"
        
        logger.info(
            "user_resolved",
            user_id=user_id,
            conversation_id=conversation_id,
            group=group,
        )
        
        return User(
            id=user_id,
            email=user_id if "@" in user_id else f"{user_id}@local",
            group_memberships=[group],
            metadata={"conversation_id": conversation_id},
        )


class ConversationStore:
    def __init__(
        self,
        host: str,
        port: int,
        user: str,
        password: str,
        database: str,
    ):
        self._host = host
        self._port = port
        self._user = user
        self._password = password
        self._database = database

    def _connect(self):
        return pymysql.connect(
            host=self._host,
            port=self._port,
            user=self._user,
            password=self._password,
            database=self._database,
            charset="utf8mb4",
            cursorclass=pymysql.cursors.DictCursor,
            autocommit=True,
        )

    def migrate(self) -> None:
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS conversation_messages (
                      id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
                      user_id VARCHAR(255) NOT NULL,
                      conversation_id VARCHAR(255) NOT NULL,
                      role ENUM('user','assistant','system','tool') NOT NULL,
                      content LONGTEXT NOT NULL,
                      created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                      PRIMARY KEY (id),
                      KEY ix_conv_messages_user_conv_created (user_id, conversation_id, created_at)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS conversation_events (
                      id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
                      user_id VARCHAR(255) NOT NULL,
                      conversation_id VARCHAR(255) NOT NULL,
                      event_type VARCHAR(100) NOT NULL,
                      event_json JSON NULL,
                      created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                      PRIMARY KEY (id),
                      KEY ix_conv_events_user_conv_created (user_id, conversation_id, created_at)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
                    """
                )
        finally:
            conn.close()

    def append_message(self, user_id: str, conversation_id: str, role: str, content: str) -> None:
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO conversation_messages (user_id, conversation_id, role, content)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (user_id, conversation_id, role, content),
                )
        finally:
            conn.close()

    def append_event(self, user_id: str, conversation_id: str, event_type: str, event_json) -> None:
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO conversation_events (user_id, conversation_id, event_type, event_json)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (user_id, conversation_id, event_type, json.dumps(event_json) if event_json is not None else None),
                )
        finally:
            conn.close()

    def get_recent_messages(self, user_id: str, conversation_id: str, limit: int) -> list[dict]:
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT role, content, created_at
                    FROM conversation_messages
                    WHERE user_id = %s AND conversation_id = %s
                    ORDER BY created_at DESC, id DESC
                    LIMIT %s
                    """,
                    (user_id, conversation_id, limit),
                )
                rows = cur.fetchall()
                return list(reversed(rows))
        finally:
            conn.close()

    def get_recent_events(self, user_id: str, conversation_id: str, limit: int) -> list[dict]:
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT event_type, event_json, created_at
                    FROM conversation_events
                    WHERE user_id = %s AND conversation_id = %s
                    ORDER BY created_at DESC, id DESC
                    LIMIT %s
                    """,
                    (user_id, conversation_id, limit),
                )
                rows = cur.fetchall()
                return list(reversed(rows))
        finally:
            conn.close()

    def clear_conversation(self, user_id: str, conversation_id: str) -> None:
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM conversation_messages WHERE user_id = %s AND conversation_id = %s",
                    (user_id, conversation_id),
                )
                cur.execute(
                    "DELETE FROM conversation_events WHERE user_id = %s AND conversation_id = %s",
                    (user_id, conversation_id),
                )
        finally:
            conn.close()


def _extract_user_and_conversation_from_request(request: Request) -> tuple[str, str]:
    user_id = request.headers.get("X-User-Id") or request.cookies.get("vanna_user") or "anonymous"
    conversation_id = (
        request.headers.get("X-Conversation-Id")
        or request.cookies.get("vanna_conversation_id")
        or "default"
    )
    return user_id, conversation_id


def _format_history(messages: list[dict]) -> str:
    parts: list[str] = []
    for m in messages:
        role = m.get("role")
        content = m.get("content")
        if not content:
            continue
        if role == "user":
            parts.append(f"User: {content}")
        elif role == "assistant":
            parts.append(f"Assistant: {content}")
        elif role == "tool":
            parts.append(f"Tool: {content}")
        else:
            parts.append(f"System: {content}")
    return "\n".join(parts).strip()


class ConversationContextMiddleware:
    """
    Pure ASGI middleware for conversation context injection.
    Avoids BaseHTTPMiddleware issues with body consumption and streaming.
    """

    def __init__(self, app, store: ConversationStore, history_limit: int, max_assistant_chars: int):
        self.app = app
        self.store = store
        self.history_limit = history_limit
        self.max_assistant_chars = max_assistant_chars

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        method = scope.get("method", "").upper()
        path = scope.get("path", "")

        if method != "POST" or path != "/api/vanna/v2/chat_sse":
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers", []))
        user_id = headers.get(b"x-user-id", b"").decode("utf-8") or "anonymous"
        conversation_id = headers.get(b"x-conversation-id", b"").decode("utf-8") or "default"

        body_parts = []
        while True:
            message = await receive()
            body_parts.append(message.get("body", b""))
            if not message.get("more_body", False):
                break

        raw_body = b"".join(body_parts)

        try:
            payload = json.loads(raw_body.decode("utf-8")) if raw_body else {}
        except Exception:
            payload = {}

        original_message = payload.get("message") if isinstance(payload, dict) else None
        new_body = raw_body

        if isinstance(original_message, str) and original_message.strip():
            try:
                self.store.append_message(
                    user_id=user_id,
                    conversation_id=conversation_id,
                    role="user",
                    content=original_message,
                )
            except Exception as e:
                logger.error("conversation_store_append_user_failed", error=str(e))

            try:
                recent = self.store.get_recent_messages(
                    user_id=user_id,
                    conversation_id=conversation_id,
                    limit=self.history_limit,
                )
            except Exception as e:
                logger.error("conversation_store_read_failed", error=str(e))
                recent = []

            history_text = _format_history(recent[:-1])
            if history_text:
                payload["message"] = (
                    "You are continuing an ongoing conversation. Use the prior messages as context.\n\n"
                    + history_text
                    + "\n\nCurrent user message: "
                    + original_message
                )
                new_body = json.dumps(payload).encode("utf-8")

        body_sent = False

        async def wrapped_receive():
            nonlocal body_sent
            if not body_sent:
                body_sent = True
                return {"type": "http.request", "body": new_body, "more_body": False}
            while True:
                msg = await receive()
                if msg["type"] == "http.disconnect":
                    return msg

        assistant_text_parts: list[str] = []

        async def wrapped_send(message):
            if message["type"] == "http.response.start":
                await send(message)
                return

            if message["type"] == "http.response.body":
                body = message.get("body", b"")

                if body:
                    try:
                        text = body.decode("utf-8") if isinstance(body, bytes) else body
                        for line in text.split("\n"):
                            if line.startswith("data:"):
                                data = line[5:].strip()
                                if data and data != "[DONE]":
                                    try:
                                        event = json.loads(data)
                                        try:
                                            event_type = event.get("type") if isinstance(event, dict) else "unknown"
                                            self.store.append_event(
                                                user_id=user_id,
                                                conversation_id=conversation_id,
                                                event_type=str(event_type) if event_type else "unknown",
                                                event_json=event,
                                            )
                                        except Exception as e:
                                            logger.error("conversation_store_append_event_failed", error=str(e))

                                        if isinstance(event, dict) and event.get("type") == "text":
                                            content = event.get("content")
                                            if isinstance(content, str) and content:
                                                assistant_text_parts.append(content)
                                    except json.JSONDecodeError:
                                        pass
                    except Exception:
                        pass

                await send(message)

                if not message.get("more_body", False):
                    assistant_text = "\n".join([p for p in assistant_text_parts if p]).strip()
                    if assistant_text:
                        try:
                            self.store.append_message(
                                user_id=user_id,
                                conversation_id=conversation_id,
                                role="assistant",
                                content=assistant_text,
                            )
                        except Exception as e:
                            logger.error("conversation_store_append_assistant_failed", error=str(e))
                return

            await send(message)

        await self.app(scope, wrapped_receive, wrapped_send)


def add_conversation_context(app, store: ConversationStore):
    history_limit = int(get_env("CONVERSATION_HISTORY_LIMIT", "12"))
    max_assistant_chars = int(get_env("CONVERSATION_ASSISTANT_MAX_CHARS", "20000"))

    @app.on_event("startup")
    async def _startup_migrate():
        store.migrate()

    app.add_middleware(
        ConversationContextMiddleware,
        store=store,
        history_limit=history_limit,
        max_assistant_chars=max_assistant_chars,
    )

    @app.get("/api/conversation/messages")
    async def get_conversation_messages(request: Request, limit: int = 50):
        user_id, conversation_id = _extract_user_and_conversation_from_request(request)
        msgs = store.get_recent_messages(user_id=user_id, conversation_id=conversation_id, limit=int(limit))
        return {
            "user_id": user_id,
            "conversation_id": conversation_id,
            "messages": [
                {
                    "role": m["role"],
                    "content": m["content"],
                    "created_at": m["created_at"].isoformat() if hasattr(m["created_at"], "isoformat") else str(m["created_at"]),
                }
                for m in msgs
            ],
        }

    @app.get("/api/conversation/events")
    async def get_conversation_events(request: Request, limit: int = 200):
        user_id, conversation_id = _extract_user_and_conversation_from_request(request)
        events = store.get_recent_events(user_id=user_id, conversation_id=conversation_id, limit=int(limit))
        normalized = []
        for e in events:
            raw = e.get("event_json")
            if isinstance(raw, (bytes, bytearray)):
                try:
                    raw = json.loads(raw.decode("utf-8"))
                except Exception:
                    raw = None
            normalized.append(
                {
                    "event_type": e.get("event_type"),
                    "event": raw,
                    "created_at": e["created_at"].isoformat() if hasattr(e["created_at"], "isoformat") else str(e["created_at"]),
                }
            )
        return {
            "user_id": user_id,
            "conversation_id": conversation_id,
            "events": normalized,
        }

    @app.delete("/api/conversation")
    async def clear_conversation(request: Request):
        user_id, conversation_id = _extract_user_and_conversation_from_request(request)
        store.clear_conversation(user_id=user_id, conversation_id=conversation_id)
        return {"status": "success", "user_id": user_id, "conversation_id": conversation_id}


# -----------------------------------------------------------------------------
# Environment Configuration
# -----------------------------------------------------------------------------
def get_required_env(name: str, fallback_names: Optional[list[str]] = None) -> str:
    """Get required environment variable with optional fallbacks."""
    value = os.getenv(name)
    if value:
        return value
    
    if fallback_names:
        for fallback in fallback_names:
            value = os.getenv(fallback)
            if value:
                return value
    
    raise ValueError(f"Required environment variable {name} not set")


def get_env(name: str, default: str, fallback_names: Optional[list[str]] = None) -> str:
    """Get environment variable with default and optional fallbacks."""
    value = os.getenv(name)
    if value:
        return value
    
    if fallback_names:
        for fallback in fallback_names:
            value = os.getenv(fallback)
            if value:
                return value
    
    return default


# -----------------------------------------------------------------------------
# Create Vanna Agent
# -----------------------------------------------------------------------------
def create_vanna_agent() -> Agent:
    """
    Create and configure Vanna AI 2.0 agent following official docs.
    
    Components:
    - OpenAILlmService: LLM for natural language understanding
    - MySQLRunner: Database connection for SQL execution
    - ChromaAgentMemory: Persistent memory for learning from successful queries
    - RunSqlTool: Tool for executing SQL queries
    - VisualizeDataTool: Tool for generating charts
    - Memory tools: For saving and searching successful patterns
    """
    
    # 1. Configure OpenAI LLM
    openai_api_key = get_required_env("OPENAI_API_KEY", ["OPENAI_API_PROJECT_KEY"])
    openai_model = get_env("OPENAI_MODEL", "gpt-4")
    
    logger.info("configuring_llm", model=openai_model)
    
    llm = OpenAILlmService(
        api_key=openai_api_key,
        model=openai_model,
    )
    
    # 2. Configure MySQL Database Runner
    mysql_host = get_required_env("MYSQL_DO_HOST", ["MYSQL_HOST"])
    mysql_port = int(get_env("MYSQL_DO_PORT", "3306", ["MYSQL_PORT"]))
    mysql_user = get_required_env("MYSQL_DO_USER", ["MYSQL_USER"])
    mysql_password = get_required_env("MYSQL_DO_PASSWORD", ["MYSQL_PASSWORD"])
    mysql_database = get_required_env("MYSQL_DO_DATABASE", ["MYSQL_DATABASE"])
    
    logger.info(
        "configuring_database",
        host=mysql_host,
        port=mysql_port,
        database=mysql_database,
    )
    
    sql_runner = MySQLRunner(
        host=mysql_host,
        port=mysql_port,
        user=mysql_user,
        password=mysql_password,
        database=mysql_database,
    )
    
    # 3. Configure ChromaDB Agent Memory (persistent learning via HTTP)
    chroma_host = get_env("CHROMA_HOST", "localhost")
    chroma_port = int(get_env("CHROMA_PORT", "8000"))
    chroma_collection = get_env("CHROMA_COLLECTION_NAME", "vanna_memories")
    
    logger.info(
        "configuring_agent_memory",
        chroma_host=chroma_host,
        chroma_port=chroma_port,
        collection=chroma_collection,
    )
    
    # HttpChromaAgentMemory connects to remote ChromaDB server
    agent_memory = HttpChromaAgentMemory(
        host=chroma_host,
        port=chroma_port,
        collection_name=chroma_collection,
    )
    
    # 4. Configure User Resolver (for PHP frontend integration)
    user_resolver = PHPFrontendUserResolver()
    
    # 5. Register Tools
    logger.info("registering_tools")
    
    tools = ToolRegistry()
    
    # RunSqlTool - executes SQL queries against MySQL
    db_tool = RunSqlTool(sql_runner=sql_runner)
    tools.register_local_tool(db_tool, access_groups=["admin", "user"])
    
    # VisualizeDataTool - generates charts from query results
    viz_tool = VisualizeDataTool()
    tools.register_local_tool(viz_tool, access_groups=["admin", "user"])
    
    # Memory tools - save successful patterns for future reference
    # SaveQuestionToolArgsTool: saves successful question->tool usage patterns
    tools.register_local_tool(
        SaveQuestionToolArgsTool(),
        access_groups=["admin"],
    )
    
    # SearchSavedCorrectToolUsesTool: searches past successful patterns
    tools.register_local_tool(
        SearchSavedCorrectToolUsesTool(),
        access_groups=["admin", "user"],
    )
    
    # SaveTextMemoryTool: saves arbitrary text memories (golden queries, notes)
    tools.register_local_tool(
        SaveTextMemoryTool(),
        access_groups=["admin", "user"],
    )
    
    # 6. Create the Vanna Agent
    logger.info("creating_agent")
    
    agent = Agent(
        llm_service=llm,
        tool_registry=tools,
        user_resolver=user_resolver,
        agent_memory=agent_memory,
    )
    
    logger.info("agent_created_successfully")
    
    return agent


# -----------------------------------------------------------------------------
# Main Entry Point
# -----------------------------------------------------------------------------
def main() -> None:
    """
    Run the Vanna AI 2.0 FastAPI server.
    
    This creates a production-ready server with:
    - POST /api/vanna/v2/chat_sse - Streaming chat endpoint
    - GET /health - Health check endpoint
    - GET /docs - OpenAPI documentation
    - Built-in web UI at root path
    """
    host = get_env("VANNA_HOST", "0.0.0.0")
    port = int(get_env("VANNA_PORT", "8000"))
    
    logger.info("starting_vanna_server", host=host, port=port)
    
    # Create the Vanna agent
    agent = create_vanna_agent()

    mysql_host = get_required_env("MYSQL_DO_HOST", ["MYSQL_HOST"])
    mysql_port = int(get_env("MYSQL_DO_PORT", "3306", ["MYSQL_PORT"]))
    mysql_user = get_required_env("MYSQL_DO_USER", ["MYSQL_USER"])
    mysql_password = get_required_env("MYSQL_DO_PASSWORD", ["MYSQL_PASSWORD"])
    mysql_database = get_required_env("MYSQL_DO_DATABASE", ["MYSQL_DATABASE"])

    conversation_store = ConversationStore(
        host=mysql_host,
        port=mysql_port,
        user=mysql_user,
        password=mysql_password,
        database=mysql_database,
    )
    
    # Create and run the FastAPI server
    server = VannaFastAPIServer(
        agent=agent,
        config={
            "cors": {
                "enabled": True,
                "allow_origins": ["*"],
            },
        },
    )
    
    # Create ASGI app and add custom training endpoints
    app = server.create_app()
    add_training_endpoints(app, agent)
    add_chat_endpoint(app, agent)
    add_conversation_context(app, conversation_store)
    
    logger.info("server_starting", host=host, port=port)
    
    # Run the server with the app
    import uvicorn
    uvicorn.run(app, host=host, port=port)


# -----------------------------------------------------------------------------
# Synchronous Chat Endpoint (Direct OpenAI call, no streaming)
# -----------------------------------------------------------------------------
def add_chat_endpoint(app, agent: Agent):
    """
    Add a synchronous REST endpoint for chat that directly calls OpenAI.
    No streaming, no SSE - just one request in, one response out.
    """
    from fastapi import HTTPException
    from pydantic import BaseModel
    import uuid
    import openai
    
    # Get OpenAI config from environment
    openai_api_key = get_required_env("OPENAI_API_KEY", ["OPENAI_API_PROJECT_KEY"])
    openai_model = get_env("OPENAI_MODEL", "gpt-4")
    
    # Get MySQL config for running SQL
    mysql_host = get_required_env("MYSQL_DO_HOST", ["MYSQL_HOST"])
    mysql_port = int(get_env("MYSQL_DO_PORT", "3306", ["MYSQL_PORT"]))
    mysql_user = get_required_env("MYSQL_DO_USER", ["MYSQL_USER"])
    mysql_password = get_required_env("MYSQL_DO_PASSWORD", ["MYSQL_PASSWORD"])
    mysql_database = get_required_env("MYSQL_DO_DATABASE", ["MYSQL_DATABASE"])
    
    # Create OpenAI client
    client = openai.OpenAI(api_key=openai_api_key)
    
    class ChatRequest(BaseModel):
        message: str
        conversation_id: str | None = None
        run_sql: bool = True  # Whether to execute generated SQL
    
    class ChatResponse(BaseModel):
        response: str
        conversation_id: str
        request_id: str
        sql: str | None = None
        data: list | None = None
        error: str | None = None
    
    def get_database_schema() -> str:
        """Get database schema for context."""
        try:
            conn = pymysql.connect(
                host=mysql_host,
                port=mysql_port,
                user=mysql_user,
                password=mysql_password,
                database=mysql_database,
            )
            cursor = conn.cursor()
            cursor.execute("SHOW TABLES")
            tables = [row[0] for row in cursor.fetchall()]
            
            schema_parts = []
            for table in tables:
                cursor.execute(f"DESCRIBE `{table}`")
                columns = cursor.fetchall()
                col_defs = [f"  {col[0]} {col[1]}" for col in columns]
                schema_parts.append(f"Table: {table}\n" + "\n".join(col_defs))
            
            conn.close()
            return "\n\n".join(schema_parts)
        except Exception as e:
            logger.error("schema_fetch_failed", error=str(e))
            return f"Error fetching schema: {str(e)}"
    
    def execute_sql(sql: str) -> tuple[list | None, str | None]:
        """Execute SQL and return results or error."""
        try:
            conn = pymysql.connect(
                host=mysql_host,
                port=mysql_port,
                user=mysql_user,
                password=mysql_password,
                database=mysql_database,
                cursorclass=pymysql.cursors.DictCursor,
            )
            cursor = conn.cursor()
            cursor.execute(sql)
            results = cursor.fetchall()
            conn.close()
            return list(results), None
        except Exception as e:
            return None, str(e)
    
    @app.post("/api/chat", response_model=ChatResponse)
    async def chat_sync(chat_request: ChatRequest):
        """
        Synchronous chat endpoint - directly calls OpenAI, no streaming.
        
        1. Gets database schema
        2. Sends user question + schema to OpenAI
        3. Extracts SQL from response
        4. Optionally executes SQL
        5. Returns complete response
        """
        try:
            conversation_id = chat_request.conversation_id or f"conv_{uuid.uuid4().hex[:8]}"
            request_id = str(uuid.uuid4())
            
            # Get database schema for context
            schema = get_database_schema()
            
            # Build the prompt
            system_prompt = f"""You are a helpful SQL assistant. You help users query a MySQL database.

Here is the database schema:
{schema}

When the user asks a question:
1. Understand what data they need
2. Generate a valid MySQL query to get that data
3. Always wrap your SQL in ```sql and ``` code blocks
4. Explain what the query does

Be concise and helpful."""

            # Call OpenAI directly (no streaming)
            response = client.chat.completions.create(
                model=openai_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": chat_request.message},
                ],
                temperature=0.1,
            )
            
            ai_response = response.choices[0].message.content
            
            # Extract SQL from response
            sql_query = None
            data = None
            sql_error = None
            
            if "```sql" in ai_response:
                # Extract SQL between ```sql and ```
                import re
                sql_match = re.search(r"```sql\s*(.*?)\s*```", ai_response, re.DOTALL)
                if sql_match:
                    sql_query = sql_match.group(1).strip()
                    
                    # Execute SQL if requested
                    if chat_request.run_sql and sql_query:
                        data, sql_error = execute_sql(sql_query)
                        if sql_error:
                            ai_response += f"\n\n**SQL Execution Error:** {sql_error}"
                        elif data:
                            ai_response += f"\n\n**Query returned {len(data)} rows.**"
            
            return ChatResponse(
                response=ai_response,
                conversation_id=conversation_id,
                request_id=request_id,
                sql=sql_query,
                data=data,
                error=sql_error,
            )
        except Exception as e:
            logger.error("chat_sync_failed", error=str(e))
            raise HTTPException(status_code=500, detail=str(e))


# -----------------------------------------------------------------------------
# Training Endpoints
# -----------------------------------------------------------------------------
def add_training_endpoints(app, agent: Agent):
    """
    Add custom endpoints for training data management.
    """
    from fastapi import HTTPException
    from fastapi.responses import JSONResponse
    import asyncio
    
    @app.post("/api/train/seed")
    async def seed_training_data():
        """
        Seed the agent memory with DDL documentation and example queries.
        This improves query accuracy by providing schema context.
        """
        try:
            training_content = get_all_training_content()
            
            # Create a mock tool context for saving memories
            from vanna.core.tool import ToolContext
            from vanna.core.user import User
            import uuid
            
            admin_user = User(
                id="system@admin",
                email="system@admin",
                group_memberships=["admin"],
                metadata={},
            )
            
            context = ToolContext(
                user=admin_user,
                agent=agent,
                agent_memory=agent.agent_memory,
                conversation_id="training-seed",
                request_id=str(uuid.uuid4()),
            )
            
            saved_count = 0
            for content in training_content:
                await agent.agent_memory.save_text_memory(
                    content=content,
                    context=context,
                )
                saved_count += 1
            
            logger.info("training_data_seeded", count=saved_count)
            
            return JSONResponse({
                "status": "success",
                "message": f"Seeded {saved_count} training memories",
                "count": saved_count,
            })
        except Exception as e:
            logger.error("training_seed_failed", error=str(e))
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/train/status")
    async def get_training_status():
        """
        Get the current training data status.
        """
        try:
            from vanna.core.tool import ToolContext
            from vanna.core.user import User
            import uuid
            
            admin_user = User(
                id="system@admin",
                email="system@admin",
                group_memberships=["admin"],
                metadata={},
            )
            
            context = ToolContext(
                user=admin_user,
                agent=agent,
                agent_memory=agent.agent_memory,
                conversation_id="training-status",
                request_id=str(uuid.uuid4()),
            )
            
            # Get recent text memories
            text_memories = await agent.agent_memory.get_recent_text_memories(
                context=context,
                limit=100,
            )
            
            # Get recent tool memories
            tool_memories = await agent.agent_memory.get_recent_memories(
                context=context,
                limit=100,
            )
            
            return JSONResponse({
                "status": "success",
                "text_memory_count": len(text_memories),
                "tool_memory_count": len(tool_memories),
            })
        except Exception as e:
            logger.error("training_status_failed", error=str(e))
            raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    main()
