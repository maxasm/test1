"""
Vanna AI 2.0 FastAPI Application
Production-ready natural language to SQL using Vanna AI 2.0 agent framework
with OpenAI LLM, MySQL database, and ChromaDB persistent memory.

Following official Vanna 2.0 docs: https://vanna.ai/docs
"""

import os
import logging
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
    
    logger.info("server_starting", host=host, port=port)
    
    # Run the server with the app
    import uvicorn
    uvicorn.run(app, host=host, port=port)


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
