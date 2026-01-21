"""
Vanna AI 2.0 FastAPI Application
Production-ready natural language to SQL conversion with persistent ChromaDB memory.
"""

import os
import json
import hashlib
import logging
from datetime import datetime
from typing import Any, Optional
from contextlib import asynccontextmanager

import chromadb
from chromadb.config import Settings
import pymysql
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
import structlog
from openai import OpenAI

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
# Pydantic Models
# -----------------------------------------------------------------------------
class ChatRequest(BaseModel):
    """Request model for /api/chat endpoint."""
    question: str = Field(..., description="Natural language question to convert to SQL")
    user: str = Field(..., description="User identifier for context isolation")
    conversation_id: str = Field(..., description="Conversation ID for context isolation")


class ChatResponse(BaseModel):
    """Response model for /api/chat endpoint."""
    answer: str = Field(..., description="Natural language answer")
    sql_query: Optional[str] = Field(None, description="Generated SQL query")
    data: Optional[list[dict[str, Any]]] = Field(None, description="Query results as JSON")
    chart: Optional[dict[str, Any]] = Field(None, description="Plotly chart configuration")
    confidence: Optional[float] = Field(None, description="Confidence score 0-1")
    from_cache: Optional[bool] = Field(None, description="Whether SQL was retrieved from memory cache")


class HealthResponse(BaseModel):
    """Response model for /health endpoint."""
    status: str
    components: dict[str, dict[str, Any]]
    timestamp: str


class MemoryItem(BaseModel):
    """Single memory item from ChromaDB."""
    id: str
    question: str
    sql_query: str
    timestamp: str
    confidence: float


class MemoryResponse(BaseModel):
    """Response model for /api/memory endpoint."""
    user: str
    conversation_id: str
    memories: list[MemoryItem]
    total_count: int


class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str
    detail: Optional[str] = None


# -----------------------------------------------------------------------------
# ChromaDB Memory Manager
# -----------------------------------------------------------------------------
class ChromaMemoryManager:
    """Manages ChromaDB for persistent agent memory with user/conversation isolation."""

    def __init__(
        self,
        host: str,
        port: int,
        openai_api_key: str,
        embedding_model: str = "text-embedding-3-small",
        collection_name: str = "vanna_memories",
    ):
        self._host = host
        self._port = port
        self._collection_name = collection_name
        self._client: Optional[chromadb.HttpClient] = None
        self._collection = None
        self._openai_client = OpenAI(api_key=openai_api_key)
        self._embedding_model = embedding_model

    def _get_client(self) -> chromadb.HttpClient:
        """Get or create ChromaDB HTTP client."""
        if self._client is None:
            self._client = chromadb.HttpClient(
                host=self._host,
                port=self._port,
                settings=Settings(anonymized_telemetry=False),
            )
        return self._client

    def _get_collection(self):
        """Get or create the memories collection."""
        if self._collection is None:
            client = self._get_client()
            self._collection = client.get_or_create_collection(
                name=self._collection_name,
                metadata={"description": "Vanna AI SQL query memories", "hnsw:space": "cosine"},
            )
        return self._collection

    def _embed_text(self, text: str) -> list[float]:
        response = self._openai_client.embeddings.create(
            model=self._embedding_model,
            input=text,
        )
        return response.data[0].embedding

    def _generate_id(self, user: str, conversation_id: str, question: str) -> str:
        """Generate unique ID for a memory entry."""
        content = f"{user}:{conversation_id}:{question}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def save_memory(
        self,
        user: str,
        conversation_id: str,
        question: str,
        sql_query: str,
        confidence: float = 1.0,
    ) -> str:
        """Save a successful SQL query to memory."""
        collection = self._get_collection()
        memory_id = self._generate_id(user, conversation_id, question)
        timestamp = datetime.utcnow().isoformat()

        embedding = self._embed_text(question)

        collection.upsert(
            ids=[memory_id],
            documents=[question],
            embeddings=[embedding],
            metadatas=[
                {
                    "user": user,
                    "conversation_id": conversation_id,
                    "sql_query": sql_query,
                    "timestamp": timestamp,
                    "confidence": confidence,
                }
            ],
        )
        logger.info(
            "memory_saved",
            memory_id=memory_id,
            user=user,
            conversation_id=conversation_id,
        )
        return memory_id

    def search_similar(
        self,
        user: str,
        conversation_id: str,
        question: str,
        n_results: int = 5,
    ) -> list[dict[str, Any]]:
        """Search for similar past queries filtered by user and conversation."""
        collection = self._get_collection()

        embedding = None
        try:
            embedding = self._embed_text(question)
        except Exception:
            embedding = None

        try:
            if embedding is not None:
                results = collection.query(
                    query_embeddings=[embedding],
                    n_results=n_results,
                    where={
                        "$and": [
                            {"user": {"$eq": user}},
                            {"conversation_id": {"$eq": conversation_id}},
                        ]
                    },
                )
            else:
                results = collection.query(
                    query_texts=[question],
                    n_results=n_results,
                    where={
                        "$and": [
                            {"user": {"$eq": user}},
                            {"conversation_id": {"$eq": conversation_id}},
                        ]
                    },
                )
        except Exception:
            if embedding is not None:
                results = collection.query(
                    query_embeddings=[embedding],
                    n_results=n_results,
                    where={"user": {"$eq": user}},
                )
            else:
                results = collection.query(
                    query_texts=[question],
                    n_results=n_results,
                    where={"user": {"$eq": user}},
                )

        memories = []
        if results and results.get("ids") and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                metadata = results["metadatas"][0][i] if results.get("metadatas") else {}
                memories.append(
                    {
                        "id": doc_id,
                        "question": results["documents"][0][i] if results.get("documents") else "",
                        "sql_query": metadata.get("sql_query", ""),
                        "timestamp": metadata.get("timestamp", ""),
                        "confidence": metadata.get("confidence", 0.0),
                        "distance": results["distances"][0][i] if results.get("distances") else 0,
                    }
                )
        return memories

    def get_memories(
        self,
        user: str,
        conversation_id: str,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Get all memories for a user/conversation pair."""
        collection = self._get_collection()

        try:
            results = collection.get(
                where={
                    "$and": [
                        {"user": {"$eq": user}},
                        {"conversation_id": {"$eq": conversation_id}},
                    ]
                },
                limit=limit,
            )
        except Exception:
            results = collection.get(
                where={"user": {"$eq": user}},
                limit=limit,
            )

        memories = []
        if results and results.get("ids"):
            for i, doc_id in enumerate(results["ids"]):
                metadata = results["metadatas"][i] if results.get("metadatas") else {}
                memories.append(
                    {
                        "id": doc_id,
                        "question": results["documents"][i] if results.get("documents") else "",
                        "sql_query": metadata.get("sql_query", ""),
                        "timestamp": metadata.get("timestamp", ""),
                        "confidence": metadata.get("confidence", 0.0),
                    }
                )
        return memories

    def health_check(self) -> dict[str, Any]:
        """Check ChromaDB connection health."""
        try:
            client = self._get_client()
            heartbeat = client.heartbeat()
            return {"status": "healthy", "heartbeat": heartbeat}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}


# -----------------------------------------------------------------------------
# MySQL Manager
# -----------------------------------------------------------------------------
class MySQLManager:
    """Manages MySQL connections and query execution."""

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

    def _get_connection(self):
        """Create a new MySQL connection."""
        return pymysql.connect(
            host=self._host,
            port=self._port,
            user=self._user,
            password=self._password,
            database=self._database,
            cursorclass=pymysql.cursors.DictCursor,
            connect_timeout=10,
        )

    def execute_query(self, sql: str) -> list[dict[str, Any]]:
        """Execute SQL query and return results as list of dicts."""
        connection = self._get_connection()
        try:
            with connection.cursor() as cursor:
                cursor.execute(sql)
                results = cursor.fetchall()
                return [dict(row) for row in results]
        finally:
            connection.close()

    def get_schema(self) -> list[dict[str, Any]]:
        """Get database schema information."""
        sql = """
        SELECT 
            TABLE_NAME, 
            COLUMN_NAME, 
            DATA_TYPE, 
            IS_NULLABLE,
            COLUMN_KEY
        FROM INFORMATION_SCHEMA.COLUMNS 
        WHERE TABLE_SCHEMA = %s
        ORDER BY TABLE_NAME, ORDINAL_POSITION
        """
        connection = self._get_connection()
        try:
            with connection.cursor() as cursor:
                cursor.execute(sql, (self._database,))
                return cursor.fetchall()
        finally:
            connection.close()

    def health_check(self) -> dict[str, Any]:
        """Check MySQL connection health."""
        try:
            connection = self._get_connection()
            connection.ping()
            connection.close()
            return {"status": "healthy", "database": self._database}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}


# -----------------------------------------------------------------------------
# Vanna AI Service
# -----------------------------------------------------------------------------
class VannaService:
    """Vanna AI 2.0 service for natural language to SQL conversion."""

    def __init__(
        self,
        openai_api_key: str,
        openai_model: str,
        mysql_manager: MySQLManager,
        memory_manager: ChromaMemoryManager,
    ):
        self._api_key = openai_api_key
        self._model = openai_model
        self._mysql = mysql_manager
        self._memory = memory_manager
        self._schema_cache: Optional[str] = None

        self._openai_client = OpenAI(api_key=openai_api_key)

    def _get_schema_context(self) -> str:
        """Get database schema as context for LLM."""
        if self._schema_cache is None:
            try:
                schema = self._mysql.get_schema()
                tables: dict[str, list[str]] = {}
                for col in schema:
                    table = col["TABLE_NAME"]
                    if table not in tables:
                        tables[table] = []
                    col_info = f"{col['COLUMN_NAME']} ({col['DATA_TYPE']})"
                    if col["COLUMN_KEY"] == "PRI":
                        col_info += " PRIMARY KEY"
                    tables[table].append(col_info)

                schema_str = "Database Schema:\n"
                for table, columns in tables.items():
                    schema_str += f"\nTable: {table}\n"
                    schema_str += "  Columns: " + ", ".join(columns) + "\n"
                self._schema_cache = schema_str
            except Exception as e:
                logger.warning("schema_fetch_failed", error=str(e))
                self._schema_cache = "Schema not available."
        return self._schema_cache

    def _build_prompt(
        self,
        question: str,
        user: str,
        conversation_id: str,
        similar_queries: list[dict[str, Any]],
    ) -> str:
        """Build the prompt for SQL generation."""
        schema = self._get_schema_context()

        prompt = f"""You are a SQL expert. Convert the user's natural language question to a valid MySQL query.

{schema}

"""
        if similar_queries:
            prompt += "Similar past queries that were successful:\n"
            for q in similar_queries[:3]:
                prompt += f"- Question: {q['question']}\n  SQL: {q['sql_query']}\n"
            prompt += "\n"

        prompt += f"""User Question: {question}

Instructions:
1. Generate ONLY the SQL query, no explanations
2. Use proper MySQL syntax
3. Be conservative with data - use LIMIT if appropriate
4. Return only SELECT queries for safety

SQL Query:"""
        return prompt

    async def generate_sql(
        self,
        question: str,
        user: str,
        conversation_id: str,
    ) -> dict[str, Any]:
        """Generate SQL from natural language question."""
        similar = self._memory.search_similar(user, conversation_id, question, n_results=5)

        if similar and similar[0].get("distance", 1) < 0.1:
            cached = similar[0]
            logger.info("cache_hit", question=question, cached_id=cached["id"])
            return {
                "sql_query": cached["sql_query"],
                "from_cache": True,
                "confidence": cached.get("confidence", 0.9),
            }

        prompt = self._build_prompt(question, user, conversation_id, similar)

        try:
            response = self._openai_client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": "You are a SQL expert. Output only valid SQL."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=500,
            )
            sql_query = response.choices[0].message.content.strip()

            sql_query = sql_query.replace("```sql", "").replace("```", "").strip()

            if not sql_query.upper().startswith("SELECT"):
                if "SELECT" in sql_query.upper():
                    idx = sql_query.upper().index("SELECT")
                    sql_query = sql_query[idx:]

            return {
                "sql_query": sql_query,
                "from_cache": False,
                "confidence": 0.8,
            }
        except Exception as e:
            logger.error("sql_generation_failed", error=str(e))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"SQL generation failed: {str(e)}",
            )

    def _generate_chart_config(
        self,
        data: list[dict[str, Any]],
        question: str,
    ) -> Optional[dict[str, Any]]:
        """Generate Plotly chart configuration if appropriate."""
        if not data or len(data) < 2:
            return None

        columns = list(data[0].keys())
        if len(columns) < 2:
            return None

        numeric_cols = []
        string_cols = []
        for col in columns:
            sample = data[0].get(col)
            if isinstance(sample, (int, float)):
                numeric_cols.append(col)
            else:
                string_cols.append(col)

        if not numeric_cols or not string_cols:
            return None

        x_col = string_cols[0]
        y_col = numeric_cols[0]

        chart_type = "bar"
        keywords = question.lower()
        if any(w in keywords for w in ["trend", "over time", "timeline", "history"]):
            chart_type = "line"
        elif any(w in keywords for w in ["distribution", "percentage", "share"]):
            chart_type = "pie"

        x_values = [row.get(x_col) for row in data]
        y_values = [row.get(y_col) for row in data]

        return {
            "data": [
                {
                    "type": chart_type,
                    "x": x_values,
                    "y": y_values,
                    "name": y_col,
                }
            ],
            "layout": {
                "title": question[:50],
                "xaxis": {"title": x_col},
                "yaxis": {"title": y_col},
            },
        }

    async def chat(
        self,
        question: str,
        user: str,
        conversation_id: str,
    ) -> ChatResponse:
        """Process a chat request and return structured response."""
        logger.info(
            "chat_request",
            question=question,
            user=user,
            conversation_id=conversation_id,
        )

        result = await self.generate_sql(question, user, conversation_id)
        sql_query = result["sql_query"]
        from_cache = result.get("from_cache", False)
        confidence = result.get("confidence", 0.8)

        data = None
        answer = ""
        chart = None

        try:
            data = self._mysql.execute_query(sql_query)

            self._memory.save_memory(
                user=user,
                conversation_id=conversation_id,
                question=question,
                sql_query=sql_query,
                confidence=min(confidence + 0.1, 1.0),
            )

            if data:
                row_count = len(data)
                answer = f"Found {row_count} result{'s' if row_count != 1 else ''}."
                chart = self._generate_chart_config(data, question)
            else:
                answer = "Query executed successfully but returned no results."

            for row in data or []:
                for key, value in row.items():
                    if isinstance(value, datetime):
                        row[key] = value.isoformat()
                    elif hasattr(value, "decode"):
                        row[key] = value.decode("utf-8", errors="replace")

        except pymysql.Error as e:
            logger.warning("sql_execution_failed", sql=sql_query, error=str(e))
            answer = f"SQL execution failed: {str(e)}"
            confidence = 0.0

        return ChatResponse(
            answer=answer,
            sql_query=sql_query,
            data=data,
            chart=chart,
            confidence=confidence,
            from_cache=from_cache,
        )

    def health_check(self) -> dict[str, Any]:
        """Check OpenAI API health."""
        try:
            self._openai_client.models.list()
            return {"status": "healthy", "model": self._model}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}


# -----------------------------------------------------------------------------
# Global instances
# -----------------------------------------------------------------------------
memory_manager: Optional[ChromaMemoryManager] = None
mysql_manager: Optional[MySQLManager] = None
vanna_service: Optional[VannaService] = None


def init_services():
    """Initialize all services from environment variables."""
    global memory_manager, mysql_manager, vanna_service

    openai_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_PROJECT_KEY")
    if not openai_key:
        raise ValueError("OPENAI_API_KEY or OPENAI_API_PROJECT_KEY required")

    openai_model = os.getenv("OPENAI_MODEL", "gpt-4")

    mysql_host = os.getenv("MYSQL_DO_HOST") or os.getenv("MYSQL_HOST")
    mysql_port = int(os.getenv("MYSQL_DO_PORT") or os.getenv("MYSQL_PORT", "3306"))
    mysql_user = os.getenv("MYSQL_DO_USER") or os.getenv("MYSQL_USER")
    mysql_password = os.getenv("MYSQL_DO_PASSWORD") or os.getenv("MYSQL_PASSWORD")
    mysql_database = os.getenv("MYSQL_DO_DATABASE") or os.getenv("MYSQL_DATABASE")

    if not all([mysql_host, mysql_user, mysql_password, mysql_database]):
        raise ValueError("MySQL configuration incomplete")

    chroma_host = os.getenv("CHROMA_HOST", "localhost")
    chroma_port = int(os.getenv("CHROMA_PORT", "8000"))
    chroma_collection = os.getenv("CHROMA_COLLECTION_NAME", "vanna_memories")
    openai_embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

    logger.info(
        "initializing_services",
        openai_model=openai_model,
        mysql_host=mysql_host,
        mysql_database=mysql_database,
        chroma_host=chroma_host,
        chroma_port=chroma_port,
    )

    memory_manager = ChromaMemoryManager(
        host=chroma_host,
        port=chroma_port,
        openai_api_key=openai_key,
        embedding_model=openai_embedding_model,
        collection_name=chroma_collection,
    )

    mysql_manager = MySQLManager(
        host=mysql_host,
        port=mysql_port,
        user=mysql_user,
        password=mysql_password,
        database=mysql_database,
    )

    vanna_service = VannaService(
        openai_api_key=openai_key,
        openai_model=openai_model,
        mysql_manager=mysql_manager,
        memory_manager=memory_manager,
    )

    logger.info("services_initialized")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    init_services()
    yield
    logger.info("application_shutdown")


# -----------------------------------------------------------------------------
# FastAPI Application
# -----------------------------------------------------------------------------
app = FastAPI(
    title="Vanna AI 2.0 API",
    description="Natural language to SQL conversion with persistent ChromaDB memory",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post(
    "/api/chat",
    response_model=ChatResponse,
    responses={500: {"model": ErrorResponse}},
    summary="Convert natural language to SQL and execute",
    description="Takes a natural language question and returns SQL query, results, and optional chart",
)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Process a natural language question and return SQL results.
    
    - **question**: Natural language question about the data
    - **user**: User identifier for memory isolation
    - **conversation_id**: Conversation ID for context isolation
    """
    if vanna_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not initialized",
        )
    return await vanna_service.chat(
        question=request.question,
        user=request.user,
        conversation_id=request.conversation_id,
    )


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check endpoint",
    description="Returns health status of all components",
)
async def health() -> HealthResponse:
    """Check health of all system components."""
    components = {}

    if memory_manager:
        components["chromadb"] = memory_manager.health_check()
    else:
        components["chromadb"] = {"status": "not_initialized"}

    if mysql_manager:
        components["mysql"] = mysql_manager.health_check()
    else:
        components["mysql"] = {"status": "not_initialized"}

    if vanna_service:
        components["openai"] = vanna_service.health_check()
    else:
        components["openai"] = {"status": "not_initialized"}

    all_healthy = all(c.get("status") == "healthy" for c in components.values())

    return HealthResponse(
        status="healthy" if all_healthy else "degraded",
        components=components,
        timestamp=datetime.utcnow().isoformat(),
    )


@app.get(
    "/api/memory/{user}/{conversation_id}",
    response_model=MemoryResponse,
    summary="Get stored memories",
    description="Retrieve all stored SQL patterns for a user/conversation pair",
)
async def get_memories(user: str, conversation_id: str) -> MemoryResponse:
    """
    Get all stored memories for a specific user and conversation.
    
    - **user**: User identifier
    - **conversation_id**: Conversation identifier
    """
    if memory_manager is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Memory service not initialized",
        )

    memories = memory_manager.get_memories(user, conversation_id)

    return MemoryResponse(
        user=user,
        conversation_id=conversation_id,
        memories=[
            MemoryItem(
                id=m["id"],
                question=m["question"],
                sql_query=m["sql_query"],
                timestamp=m["timestamp"],
                confidence=m["confidence"],
            )
            for m in memories
        ],
        total_count=len(memories),
    )


@app.get("/", summary="Root endpoint", include_in_schema=False)
async def root():
    """Root endpoint with API info."""
    return {
        "name": "Vanna AI 2.0 API",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/health",
    }


# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------
def main() -> None:
    """Run the FastAPI application."""
    host = os.getenv("VANNA_HOST", "0.0.0.0")
    port = int(os.getenv("VANNA_PORT", "8000"))

    logger.info("starting_server", host=host, port=port)

    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=os.getenv("APP_ENV", "production") == "development",
        log_level=LOG_LEVEL.lower(),
    )


if __name__ == "__main__":
    main()
