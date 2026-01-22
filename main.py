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

from training_data import get_all_training_content


# -----------------------------------------------------------------------------
# Custom MySQL Runner with SSL Support
# -----------------------------------------------------------------------------
class SSLMySQLRunner(MySQLRunner):
    """
    MySQLRunner subclass that adds SSL support for secure database connections.
    
    DigitalOcean managed databases require SSL connections.
    """
    
    def __init__(
        self,
        host: str,
        port: int,
        user: str,
        password: str,
        database: str,
        ssl_ca: str | None = None,
    ):
        # Store SSL config before calling parent
        self._ssl_ca = ssl_ca
        self._ssl_config = {"ca": ssl_ca} if ssl_ca else None
        
        # Store connection params for our custom connection method
        self._host = host
        self._port = port
        self._user = user
        self._password = password
        self._database = database
        
        # Call parent init
        super().__init__(
            host=host,
            port=port,
            user=user,
            password=password,
            database=database,
        )
    
    def _get_connection(self):
        """Override to add SSL support to the connection."""
        if self._ssl_ca and not os.path.exists(self._ssl_ca):
            raise FileNotFoundError(
                f"MYSQL_DO_SSL_CA file not found at '{self._ssl_ca}'. "
                "Make sure it is mounted into the container at this path."
            )
        return pymysql.connect(
            host=self._host,
            port=self._port,
            user=self._user,
            password=self._password,
            database=self._database,
            ssl=self._ssl_config,
            cursorclass=pymysql.cursors.DictCursor,
            charset="utf8mb4",
        )
    
    def run_sql(self, sql, *args, **kwargs) -> list[dict]:
        """Execute SQL with SSL connection.

        Accepts *args/**kwargs for compatibility with the upstream Vanna tool interface.
        """
        sql_text = sql
        try:
            if isinstance(sql_text, str):
                pass
            elif hasattr(sql_text, "sql"):
                sql_text = getattr(sql_text, "sql")
            elif isinstance(sql_text, dict) and "sql" in sql_text:
                sql_text = sql_text.get("sql")
            else:
                sql_text = str(sql_text)
        except Exception:
            sql_text = str(sql)

        try:
            conn = self._get_connection()
        except Exception as e:
            logger.error(
                "mysql_connect_failed",
                host=self._host,
                port=self._port,
                database=self._database,
                user=self._user,
                ssl_ca=self._ssl_ca,
                error=str(e),
            )
            raise

        try:
            with conn.cursor() as cursor:
                cursor.execute(sql_text)
                results = cursor.fetchall()
                return list(results)
        except Exception as e:
            logger.error(
                "mysql_query_failed",
                host=self._host,
                port=self._port,
                database=self._database,
                user=self._user,
                ssl_ca=self._ssl_ca,
                sql=sql_text,
                error=str(e),
            )
            raise
        finally:
            try:
                conn.close()
            except Exception:
                pass


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
    mysql_ssl_ca = os.getenv("MYSQL_DO_SSL_CA")
    
    logger.info(
        "configuring_database",
        host=mysql_host,
        port=mysql_port,
        database=mysql_database,
        ssl_enabled=bool(mysql_ssl_ca),
    )
    
    # Use custom SSL-enabled MySQL runner
    sql_runner = SSLMySQLRunner(
        host=mysql_host,
        port=mysql_port,
        user=mysql_user,
        password=mysql_password,
        database=mysql_database,
        ssl_ca=mysql_ssl_ca,
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
    
    # Add API metadata for documentation
    app.title = "Vanna AI API"
    app.description = """
## Natural Language to SQL API

This API allows you to query your MySQL database using natural language questions.

### Features
- **Chat**: Ask questions in plain English and get SQL queries + results
- **Training**: Seed the AI with schema documentation for better accuracy
- **Streaming**: SSE endpoint for real-time responses (via Vanna framework)

### Authentication
Pass user identity via headers:
- `X-User-Id`: User identifier
- `X-Conversation-Id`: Optional conversation tracking ID
"""
    app.version = "2.0.0"
    app.openapi_tags = [
        {"name": "Chat", "description": "Natural language to SQL chat endpoints"},
        {"name": "Training", "description": "Agent memory and training data management"},
    ]
    
    add_training_endpoints(app, agent)
    add_chat_endpoint(app, agent)
    
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
    from pydantic import BaseModel, Field
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
    mysql_ssl_ca = os.getenv("MYSQL_DO_SSL_CA")
    mysql_ssl_config = {"ca": mysql_ssl_ca} if mysql_ssl_ca else None
    
    # Create OpenAI client
    client = openai.OpenAI(api_key=openai_api_key)
    
    class ChatRequest(BaseModel):
        """Request body for the chat endpoint."""
        message: str = Field(
            ...,
            description="Natural language question about the database",
            examples=["Show me all tables", "How many users signed up last month?"]
        )
        conversation_id: str | None = Field(
            default=None,
            description="Optional conversation ID for tracking. Auto-generated if not provided."
        )
        run_sql: bool = Field(
            default=True,
            description="Whether to execute the generated SQL query against the database"
        )
    
    class ChatResponse(BaseModel):
        """Response from the chat endpoint."""
        response: str = Field(
            ...,
            description="AI-generated response with explanation and SQL query"
        )
        conversation_id: str = Field(
            ...,
            description="Conversation ID for this chat session"
        )
        request_id: str = Field(
            ...,
            description="Unique identifier for this request"
        )
        sql: str | None = Field(
            default=None,
            description="Extracted SQL query from the AI response, if any"
        )
        data: list | None = Field(
            default=None,
            description="Query results as a list of dictionaries, if SQL was executed"
        )
        error: str | None = Field(
            default=None,
            description="Error message if SQL execution failed"
        )
    
    def get_database_schema() -> str:
        """Get database schema for context."""
        try:
            conn = pymysql.connect(
                host=mysql_host,
                port=mysql_port,
                user=mysql_user,
                password=mysql_password,
                database=mysql_database,
                ssl=mysql_ssl_config,
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
                ssl=mysql_ssl_config,
                cursorclass=pymysql.cursors.DictCursor,
            )
            cursor = conn.cursor()
            cursor.execute(sql)
            results = cursor.fetchall()
            conn.close()
            return list(results), None
        except Exception as e:
            return None, str(e)
    
    @app.post("/api/chat", response_model=ChatResponse, tags=["Chat"])
    async def chat_sync(chat_request: ChatRequest):
        """
        Natural language to SQL chat endpoint.
        
        Send a natural language question about your database and receive:
        - An AI-generated explanation
        - The SQL query to answer your question
        - Optionally, the query results (if run_sql is true)
        
        **Example questions:**
        - "Show me all tables in the database"
        - "How many orders were placed last month?"
        - "What are the top 10 customers by revenue?"
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
                        else:
                            row_count = len(data or [])
                            if row_count == 0:
                                ai_response = "No results were found for your question."
                            else:
                                # Summarize the results in natural language so the user doesn't need to run SQL.
                                # Keep the SQL in the dedicated response field.
                                max_rows = 25
                                preview_rows = (data or [])[:max_rows]
                                summary_prompt = (
                                    "You are a helpful data analyst. Answer the user's question using ONLY the SQL results provided. "
                                    "Do not include SQL in your answer. If the results don't contain enough information, say so. "
                                    "Be concise.\n\n"
                                    f"User question: {chat_request.message}\n"
                                    f"Row count: {row_count}\n"
                                    f"Rows (up to {max_rows}): {json.dumps(preview_rows, ensure_ascii=False)}\n"
                                )

                                try:
                                    summary = client.chat.completions.create(
                                        model=openai_model,
                                        messages=[
                                            {"role": "system", "content": "Answer using the provided rows."},
                                            {"role": "user", "content": summary_prompt},
                                        ],
                                        temperature=0.1,
                                    )
                                    summarized_text = summary.choices[0].message.content
                                    if summarized_text and summarized_text.strip():
                                        ai_response = summarized_text.strip()
                                except Exception as e:
                                    logger.error("chat_result_summarize_failed", error=str(e))
                                    ai_response = f"Query returned {row_count} rows."
            
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
    from pydantic import BaseModel, Field
    import asyncio
    
    class TrainingSeedResponse(BaseModel):
        """Response from the training seed endpoint."""
        status: str = Field(..., description="Status of the operation (success/error)")
        message: str = Field(..., description="Human-readable message about the operation")
        count: int = Field(..., description="Number of training memories seeded")
    
    class TrainingStatusResponse(BaseModel):
        """Response from the training status endpoint."""
        status: str = Field(..., description="Status of the operation (success/error)")
        text_memory_count: int = Field(..., description="Number of text memories stored")
        tool_memory_count: int = Field(..., description="Number of tool usage memories stored")
    
    @app.post("/api/train/seed", response_model=TrainingSeedResponse, tags=["Training"])
    async def seed_training_data():
        """
        Seed the agent memory with DDL documentation and example queries.
        
        This endpoint loads predefined training data (schema documentation, 
        example queries, business rules) into the agent's memory to improve 
        SQL generation accuracy.
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
    
    @app.get("/api/train/status", response_model=TrainingStatusResponse, tags=["Training"])
    async def get_training_status():
        """
        Get the current training data status.
        
        Returns the count of text memories and tool usage memories 
        currently stored in the agent's memory.
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
