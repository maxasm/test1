"""
Vanna AI 2.0 FastAPI Application
Production-ready natural language to SQL using Vanna AI 2.0 agent framework
with OpenAI LLM, MySQL database, and ChromaDB persistent memory.

Following official Vanna 2.0 docs: https://vanna.ai/docs
"""

import os
import logging
import json
import time
import hmac
import hashlib
import re
from datetime import datetime, date
from decimal import Decimal
from typing import Optional, Dict, List, Set

from dotenv import load_dotenv
import structlog


# -----------------------------------------------------------------------------
# Custom JSON Encoder for Database Types
# -----------------------------------------------------------------------------
class DatabaseJSONEncoder(json.JSONEncoder):
    """
    Custom JSON encoder that handles database-specific types:
    - Decimal -> float
    - datetime -> ISO format string
    - date -> ISO format string
    """
    
    def default(self, obj):
        if isinstance(obj, Decimal):
            # Convert Decimal to float for JSON serialization
            return float(obj)
        elif isinstance(obj, datetime):
            # Convert datetime to ISO format string
            return obj.isoformat()
        elif isinstance(obj, date):
            # Convert date to ISO format string
            return obj.isoformat()
        # Let the base class default method raise the TypeError
        return super().default(obj)

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
from vanna import Agent, AgentConfig
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
import asyncio

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
    
    def _normalize_sql(self, sql) -> str:
        sql_text = sql
        try:
            if isinstance(sql_text, str):
                return sql_text
            if hasattr(sql_text, "sql"):
                return getattr(sql_text, "sql")
            if isinstance(sql_text, dict) and "sql" in sql_text:
                return str(sql_text.get("sql"))
            return str(sql_text)
        except Exception:
            return str(sql)

    def _run_sql_sync(self, sql_text: str) -> list[dict]:
        """Blocking SQL execution. Use only from a worker thread."""
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
            if isinstance(e, FileNotFoundError):
                raise RuntimeError(
                    f"MySQL SSL CA file not found: {e}. "
                    "If you set MYSQL_DO_SSL_CA, ensure the file exists on the host and is mounted into the container at that exact path."
                ) from e
            raise RuntimeError(
                f"MySQL connection failed (host={self._host}, port={self._port}, database={self._database}, ssl={'on' if self._ssl_ca else 'off'}): {e}"
            ) from e

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
            raise RuntimeError(f"MySQL query failed: {e}") from e
        finally:
            try:
                conn.close()
            except Exception:
                pass

    async def run_sql(self, sql, *args, **kwargs) -> list[dict]:
        """Execute SQL with SSL connection (awaitable).

        Vanna's tool layer may `await` this call. We run the blocking PyMySQL
        work in a thread via asyncio.to_thread().
        """
        sql_text = self._normalize_sql(sql)

        logger.info(
            "mysql_run_sql_called",
            sql_type=type(sql).__name__,
            args_len=len(args),
            kwargs_keys=sorted(list(kwargs.keys())),
            sql_preview=(sql_text[:200] if isinstance(sql_text, str) else str(sql_text)[:200]),
        )

        return await asyncio.to_thread(self._run_sql_sync, sql_text)


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
            logger.info(
                "chromadb_client_creating",
                host=self.host,
                port=self.port,
                collection_name=self.collection_name,
            )
            self._client = chromadb.HttpClient(
                host=self.host,
                port=self.port,
                settings=Settings(anonymized_telemetry=False),
            )
            logger.info("chromadb_client_created")
        return self._client

    def _get_collection(self):
        """Get or create the configured Chroma collection."""
        if self._collection is None:
            client = self._get_client()
            logger.info(
                "chromadb_collection_get_or_create",
                collection_name=self.collection_name,
            )
            self._collection = client.get_or_create_collection(
                name=self.collection_name,
            )
            logger.info(
                "chromadb_collection_ready",
                collection_name=self.collection_name,
            )
        return self._collection

    def raw_get(
        self,
        *,
        limit: int = 100,
        offset: int = 0,
        where: dict | None = None,
        where_document: dict | None = None,
        include: list[str] | None = None,
    ) -> dict:
        logger.debug(
            "chromadb_raw_get",
            limit=limit,
            offset=offset,
            where=where,
            where_document=where_document,
            include=include,
        )
        collection = self._get_collection()
        payload = {
            "limit": limit,
            "offset": offset,
            "include": include or ["documents", "metadatas"],
        }
        if where is not None:
            payload["where"] = where
        if where_document is not None:
            payload["where_document"] = where_document
        
        result = collection.get(**payload)
        logger.debug(
            "chromadb_raw_get_result",
            result_count=len(result.get("ids", [])),
        )
        return result

    def raw_delete(self, *, ids: list[str]) -> None:
        logger.info("chromadb_raw_delete", ids=ids)
        collection = self._get_collection()
        collection.delete(ids=ids)
        logger.debug("chromadb_raw_delete_completed", ids_count=len(ids))


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
# Signed-header resolver (RequestContext-based) - from sample.py
# -----------------------------------------------------------------------------
class SignedHeaderUserResolver(UserResolver):
    """
    Headers expected (set by PHP proxy or curl):
      X-User-Email
      X-User-Id
      X-User-Groups
      X-Auth-Ts
      X-Auth-Nonce
      X-Auth-Signature  (HMAC-SHA256 over: ts\\nnonce\\nemail\\nuid\\ngroups_raw)
    """

    def __init__(
        self,
        secret: str,
        max_skew_seconds: int = 300,
        nonce_ttl_seconds: int = 900,
        max_nonce_cache: int = 10000,
    ):
        self.secret = secret.encode("utf-8") if secret else b""
        self.max_skew = int(max_skew_seconds)
        self.nonce_ttl = int(nonce_ttl_seconds)
        self.max_nonce_cache = int(max_nonce_cache)
        self._nonces: Dict[str, float] = {}

    def _cleanup_nonces(self) -> None:
        now = time.time()
        expired = [k for k, exp in self._nonces.items() if exp <= now]
        for k in expired:
            self._nonces.pop(k, None)
        # Cap cache (drop oldest)
        if len(self._nonces) > self.max_nonce_cache:
            for k, _ in sorted(self._nonces.items(), key=lambda kv: kv[1])[
                : len(self._nonces) - self.max_nonce_cache
            ]:
                self._nonces.pop(k, None)

    def _check_replay(self, nonce: str) -> None:
        self._cleanup_nonces()
        if nonce in self._nonces:
            raise ValueError("Replay detected (nonce already used).")
        self._nonces[nonce] = time.time() + self.nonce_ttl

    async def resolve_user(self, request_context: RequestContext) -> User:
        if not self.secret:
            raise ValueError("Missing VANNA_PROXY_HMAC_SECRET (fail closed).")

        email = (request_context.get_header("X-User-Email") or "").strip().lower()
        uid = (request_context.get_header("X-User-Id") or "").strip()
        groups_raw = (request_context.get_header("X-User-Groups") or "").strip()
        ts_s = (request_context.get_header("X-Auth-Ts") or "").strip()
        nonce = (request_context.get_header("X-Auth-Nonce") or "").strip()
        sig = (request_context.get_header("X-Auth-Signature") or "").strip()

        if not (email and uid and groups_raw and ts_s and nonce and sig):
            raise ValueError("Missing signed identity headers.")

        try:
            ts = int(ts_s)
        except ValueError:
            raise ValueError("Invalid X-Auth-Ts.")

        now = int(time.time())
        skew = abs(now - ts)
        if skew > self.max_skew:
            raise ValueError(
                f"Signature timestamp outside allowed skew "
                f"(skew={skew}s, max_skew={self.max_skew}s, server_now={now}, ts={ts}, ts_raw={ts_s!r})"
            )

        self._check_replay(nonce)

        base = f"{ts}\n{nonce}\n{email}\n{uid}\n{groups_raw}"
        expected = hmac.new(self.secret, base.encode("utf-8"), hashlib.sha256).hexdigest()

        if not hmac.compare_digest(expected, sig):
            raise ValueError("Invalid signature.")

        groups: List[str] = [g.strip() for g in groups_raw.split(",") if g.strip()] or ["users"]
        logger.warning("RESOLVER HIT email=%r uid=%r groups=%r", email, uid, groups)
        return User(id=uid, email=email, group_memberships=groups)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _csv_to_set(value: Optional[str]) -> Set[str]:
    if not value:
        return set()
    return {v.strip().lower() for v in value.split(",") if v.strip()}


# -----------------------------------------------------------------------------
# Safe SQL tool wrapper (Vanna 2.x expects RunSqlTool.run)
# -----------------------------------------------------------------------------
class SafeRunSqlTool(RunSqlTool):
    """
    Safe SQL tool wrapper with security features.
    Combine with:
    - DB user with SELECT-only permissions on appropriate views/tables
    - Views that expose only allowed columns
    """

    DEFAULT_LIMIT = int(os.getenv("AI_DEFAULT_LIMIT", "200"))
    MAX_LIMIT = int(os.getenv("AI_MAX_LIMIT", "500"))

    # Optional deny lists from env
    BLOCKED_RELATIONS = _csv_to_set(os.getenv("AI_BLOCKED_RELATIONS"))
    BLOCKED_COLS = _csv_to_set(os.getenv("AI_BLOCKED_COLS"))
    
    # Allowed tables/views (can be configured via environment)
    ALLOWED_TABLES = _csv_to_set(os.getenv("AI_ALLOWED_TABLES", ""))

    def run(self, sql: str, user=None):
        user_id = getattr(user, "id", None) if user else None
        logger.info(
            "safe_sql_tool_execution_started",
            user_id=user_id,
            sql_preview=(sql or "")[:200],
            sql_length=len(sql or ""),
        )

        s = (sql or "").strip()
        s_low = s.lower()

        # single statement only
        if ";" in s_low.rstrip(";"):
            logger.warning("safe_sql_tool_rejected_multiple_statements", user_id=user_id)
            raise ValueError("Multiple statements are not allowed.")

        # only SELECT/WITH
        if not (s_low.startswith("select") or s_low.startswith("with")):
            logger.warning("safe_sql_tool_rejected_non_select", user_id=user_id, sql_type=s_low.split()[0] if s_low else "empty")
            raise ValueError("Only SELECT/WITH queries are allowed.")

        # Check for allowed tables if configured
        if self.ALLOWED_TABLES:
            table_found = False
            for table in self.ALLOWED_TABLES:
                if re.search(rf"\b{re.escape(table)}\b", s_low):
                    table_found = True
                    break
            if not table_found:
                allowed_list = ", ".join(sorted(self.ALLOWED_TABLES))
                logger.warning(
                    "safe_sql_tool_rejected_table_not_allowed",
                    user_id=user_id,
                    allowed_tables=list(self.ALLOWED_TABLES),
                )
                raise ValueError(f"Query must use one of the allowed tables/views: {allowed_list}")

        # no SELECT *
        if re.search(r"\bselect\s+\*\b", s_low):
            logger.warning("safe_sql_tool_rejected_select_star", user_id=user_id)
            raise ValueError("SELECT * is not allowed.")

        # blocked relations (optional)
        for rel in self.BLOCKED_RELATIONS:
            if re.search(rf"\b{re.escape(rel)}\b", s_low):
                logger.warning(
                    "safe_sql_tool_rejected_blocked_relation",
                    user_id=user_id,
                    blocked_relation=rel,
                )
                raise ValueError(f"Relation '{rel}' is not allowed.")

        # blocked columns (optional)
        for col in self.BLOCKED_COLS:
            if re.search(rf"\b{re.escape(col)}\b", s_low):
                logger.warning(
                    "safe_sql_tool_rejected_blocked_column",
                    user_id=user_id,
                    blocked_column=col,
                )
                raise ValueError(f"Column '{col}' is not allowed.")

        # enforce LIMIT (and cap)
        m = re.search(r"\blimit\s+(\d+)\b", s_low)
        if not m:
            s = s.rstrip() + f" LIMIT {self.DEFAULT_LIMIT}"
            logger.debug(
                "safe_sql_tool_added_limit",
                user_id=user_id,
                added_limit=self.DEFAULT_LIMIT,
            )
        else:
            lim = int(m.group(1))
            if lim > self.MAX_LIMIT:
                s = re.sub(r"(?i)\blimit\s+\d+\b", f"LIMIT {self.MAX_LIMIT}", s, count=1)
                logger.debug(
                    "safe_sql_tool_capped_limit",
                    user_id=user_id,
                    original_limit=lim,
                    capped_limit=self.MAX_LIMIT,
                )

        logger.debug(
            "safe_sql_tool_passing_to_parent",
            user_id=user_id,
            final_sql_preview=s[:200],
        )
        
        try:
            result = super().run(s, user=user)
            logger.info(
                "safe_sql_tool_execution_success",
                user_id=user_id,
                result_row_count=len(result) if result else 0,
            )
            return result
        except Exception as e:
            logger.error(
                "safe_sql_tool_execution_failed",
                user_id=user_id,
                error=str(e),
                error_type=type(e).__name__,
            )
            raise


# -----------------------------------------------------------------------------
# Memory seeding
# -----------------------------------------------------------------------------
SCHEMA_DOC = """
Database Query Guidelines (MySQL):

General Principles:
- Connect to the actual MySQL database to discover available tables and columns
- Use appropriate tables and columns based on the actual database schema
- Always verify table and column names exist before using them in queries

Query Rules:
- Only use SELECT or WITH queries (no data modification)
- Never use SELECT *; always specify explicit column names
- Always add a LIMIT clause (default 200 rows unless otherwise specified)
- Use appropriate table and column names based on the actual database schema
- For text filters: use robust UPPER(TRIM(col)) LIKE '%TERM%' pattern matching
- Date/time filters should use appropriate database functions

Best Practices:
- Understand the database schema before generating queries
- Use appropriate joins based on foreign key relationships
- Aggregate functions (SUM, COUNT, AVG) should be used with GROUP BY when needed
- Always consider performance implications of queries
- Use parameterized queries or proper escaping to prevent SQL injection

Output Format:
- Provide clear column aliases for better readability
- Format dates consistently (YYYY-MM-DD)
- Include appropriate WHERE clauses to filter data meaningfully
- Structure results in a logical, readable format

Important:
- Data always comes from the connected MySQL database
- The system should dynamically discover available tables and columns
- Use English table and column names in queries
- Follow standard SQL conventions for the specific database
""".strip()


def seed_memory(agent_memory, text: str, meta: dict) -> bool:
    for fn_name in ("add_text", "add_document", "add_documents", "upsert_text", "upsert"):
        fn = getattr(agent_memory, fn_name, None)
        if not fn:
            continue
        try:
            fn(text, metadata=meta)
            return True
        except TypeError:
            try:
                fn([text], metadatas=[meta])
                return True
            except Exception:
                pass
        except Exception:
            pass
    return False


GOLDEN_QUERIES = [
    (
        """
Example: Get recent records from a table
SELECT 
    id,
    created_at,
    name,
    status,
    amount
FROM your_table_name
WHERE created_at >= DATE_SUB(CURDATE(), INTERVAL 30 DAY)
ORDER BY created_at DESC
LIMIT 50;
""".strip(),
        {"type": "example_query", "topic": "recent_records"},
    ),
    (
        """
Example: Get details for a specific record
SELECT 
    id,
    name,
    description,
    quantity,
    price,
    total_amount
FROM your_table_name
WHERE id = 123456
ORDER BY id ASC
LIMIT 200;
""".strip(),
        {"type": "example_query", "topic": "record_detail"},
    ),
    (
        """
Example: Search for records by keyword
SELECT 
    id,
    name,
    category,
    description,
    created_at
FROM your_table_name
WHERE 
    UPPER(TRIM(name)) LIKE '%SEARCH_TERM%'
    OR UPPER(TRIM(description)) LIKE '%SEARCH_TERM%'
    OR UPPER(TRIM(category)) LIKE '%SEARCH_TERM%'
ORDER BY created_at DESC
LIMIT 200;
""".strip(),
        {"type": "example_query", "topic": "keyword_search"},
    ),
    (
        """
Example: Aggregate data by category
SELECT 
    category,
    COUNT(*) AS record_count,
    ROUND(SUM(amount), 2) AS total_amount,
    ROUND(AVG(amount), 2) AS average_amount
FROM your_table_name
WHERE created_at >= DATE_SUB(CURDATE(), INTERVAL 12 MONTH)
GROUP BY category
ORDER BY total_amount DESC
LIMIT 50;
""".strip(),
        {"type": "example_query", "topic": "aggregate_by_category"},
    ),
]





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
    
    logger.info(
        "configuring_llm",
        model=openai_model,
        api_key_configured=bool(openai_api_key and len(openai_api_key) > 10),
    )
    
    llm = OpenAILlmService(
        api_key=openai_api_key,
        model=openai_model,
    )
    
    logger.debug("llm_service_created", llm_service_type=type(llm).__name__)
    
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
    
    # 4. Configure User Resolver (choose based on environment)
    use_signed_auth = os.getenv("VANNA_PROXY_HMAC_SECRET", "").strip() != ""
    if use_signed_auth:
        logger.info("using_signed_header_auth")
        user_resolver = SignedHeaderUserResolver(
            secret=os.getenv("VANNA_PROXY_HMAC_SECRET", ""),
            max_skew_seconds=int(os.getenv("VANNA_SIG_MAX_SKEW", "3600")),
            nonce_ttl_seconds=int(os.getenv("VANNA_NONCE_TTL", "900")),
            max_nonce_cache=int(os.getenv("VANNA_NONCE_CACHE", "10000")),
        )
    else:
        logger.info("using_php_frontend_auth")
        user_resolver = PHPFrontendUserResolver()
    
    # 5. Register Tools
    logger.info("registering_tools")
    
    tools = ToolRegistry()
    
    # Choose between SafeRunSqlTool and regular RunSqlTool based on environment
    use_safe_sql = os.getenv("AI_ENABLE_SAFE_SQL", "true").lower() == "true"
    if use_safe_sql:
        logger.info("using_safe_sql_tool")
        db_tool = SafeRunSqlTool(sql_runner=sql_runner)
    else:
        logger.info("using_regular_sql_tool")
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
    
    # 6. Seed memory with schema and golden queries (if enabled)
    seed_enabled = os.getenv("CHROMA_SEED_ENABLED", "true").lower() == "true"
    if seed_enabled:
        seed_version = int(os.getenv("CHROMA_SEED_VERSION", "1"))
        persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_memory")
        marker_path = os.path.join(persist_dir, f".seeded_v{seed_version}")
        
        if not os.path.exists(marker_path):
            logger.info("seeding_memory", version=seed_version)
            seed_memory(
                agent_memory,
                SCHEMA_DOC,
                {"type": "db_schema", "scope": "general_database", "version": seed_version},
            )
            for text, meta in GOLDEN_QUERIES:
                seed_memory(
                    agent_memory,
                    text.strip(),
                    {"scope": "example_queries", "version": seed_version, **meta},
                )
            try:
                os.makedirs(persist_dir, exist_ok=True)
                with open(marker_path, "w", encoding="utf-8") as f:
                    f.write(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))
                logger.info("memory_seeded", marker=marker_path)
            except Exception as e:
                logger.error("memory_seed_marker_failed", error=str(e))
        else:
            logger.info("memory_already_seeded", version=seed_version)
    
    # 7. Configure Agent with tool usage reinforcement
    logger.info("configuring_agent_with_tool_prompt")
    
    # Agent config: reinforce tool usage and output-first behavior
    cfg = AgentConfig()
    force_txt = """
You have a SQL tool named run_sql that executes real MySQL SELECT/WITH queries.
Use appropriate tables and views based on the database schema.

WORKFLOW (output-first):
1) If the question is database-related, immediately execute a first SELECT via run_sql.
2) If the user doesn't specify a time period: use appropriate default filters based on the context.
3) Always show results (even if 0 rows: say it's 0 and give 1 suggestion to refine).
4) Ask at most 1 clarification question, and only if it's truly ambiguous.

SQL RULES:
- Only SELECT or WITH (no mutations).
- Never SELECT *; choose explicit columns.
- Always add a LIMIT (max 200 unless requested).
- For text filters: use robust UPPER(TRIM(col)) LIKE '%TERM%'.
- Use appropriate table/column names based on the database schema.

OUTPUT:
- Give a brief explanation + table result.
""".strip()

    for attr in ("system_prompt", "instructions", "prompt", "agent_instructions"):
        if hasattr(cfg, attr):
            try:
                setattr(cfg, attr, force_txt)
            except Exception:
                pass
    
    # 8. Create the Vanna Agent
    logger.info("creating_agent")
    
    agent = Agent(
        llm_service=llm,
        tool_registry=tools,
        user_resolver=user_resolver,
        agent_memory=agent_memory,
        config=cfg,
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
    add_memory_admin_endpoints(app, agent)
    add_chat_endpoint(app, agent)
    add_chat_sse_endpoint(app)
    
    # Add debug endpoints
    add_debug_endpoints(app, agent)
    
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

    def normalize_generated_sql(sql: str) -> str:
        """Normalize LLM-generated SQL to avoid placeholders and improve MySQL compatibility."""
        if not sql:
            return sql

        normalized = sql.strip()
        normalized = normalized.replace("your_database_name", mysql_database)
        normalized = normalized.replace("<database_name>", mysql_database)
        normalized = normalized.replace("database_name", mysql_database)

        import re

        normalized = re.sub(
            r"(?is)^\\s*SHOW\\s+TABLES\\s+(IN|FROM)\\s+`?" + re.escape(mysql_database) + r"`?\\s*;?\\s*$",
            "SHOW TABLES;",
            normalized,
        )
        normalized = re.sub(
            r"(?is)^\\s*SHOW\\s+TABLES\\s+(IN|FROM)\\s+`?your_database_name`?\\s*;?\\s*$",
            "SHOW TABLES;",
            normalized,
        )
        return normalized
    
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

    def normalize_generated_sql(sql: str) -> str:
        """Normalize LLM-generated SQL to avoid placeholders and improve MySQL compatibility."""
        if not sql:
            return sql

        normalized = sql.strip()
        # Common placeholder patterns from LLMs
        normalized = normalized.replace("your_database_name", mysql_database)
        normalized = normalized.replace("<database_name>", mysql_database)
        normalized = normalized.replace("database_name", mysql_database)

        # In MySQL, SHOW TABLES; lists tables for the current database.
        # Some models generate SHOW TABLES IN/FROM <db>; which can introduce placeholders.
        import re

        normalized = re.sub(
            r"(?is)^\\s*SHOW\\s+TABLES\\s+(IN|FROM)\\s+`?" + re.escape(mysql_database) + r"`?\\s*;?\\s*$",
            "SHOW TABLES;",
            normalized,
        )
        normalized = re.sub(
            r"(?is)^\\s*SHOW\\s+TABLES\\s+(IN|FROM)\\s+`?your_database_name`?\\s*;?\\s*$",
            "SHOW TABLES;",
            normalized,
        )

        return normalized
    
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
            
            logger.info(
                "chat_sync_request_received",
                conversation_id=conversation_id,
                request_id=request_id,
                message_preview=chat_request.message[:100],
                run_sql=chat_request.run_sql,
            )
            
            # Get database schema for context
            schema = get_database_schema()
            
            logger.debug(
                "database_schema_retrieved",
                schema_length=len(schema),
                has_tables="Table:" in schema,
            )
            
            # Build the prompt
            system_prompt = f"""You are a helpful SQL assistant. You help users query a MySQL database.

The current database name is: {mysql_database}

Here is the database schema:
{schema}

When the user asks a question:
1. Understand what data they need
2. Generate a valid MySQL query to get that data
3. Always wrap your SQL in ```sql and ``` code blocks
4. Explain what the query does

Important:
- Do NOT use placeholders like 'your_database_name'.
- Use the current database implicitly (e.g. use `SHOW TABLES;` instead of `SHOW TABLES IN ...`).

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
                    sql_query = normalize_generated_sql(sql_match.group(1).strip())
                    
                    # Execute SQL if requested using agent's tool registry
                    if chat_request.run_sql and sql_query:
                        try:
                            # Get the SQL tool from agent's tool registry
                            # First try to find SafeRunSqlTool, then RunSqlTool
                            sql_tool = None
                            tools = agent.tool_registry
                            
                            # Try different ways to access tools
                            for attr in ["_tools", "tools", "_registered_tools", "registered_tools", "local_tools"]:
                                if hasattr(tools, attr):
                                    tool_dict = getattr(tools, attr)
                                    if isinstance(tool_dict, dict):
                                        for tool_name, tool in tool_dict.items():
                                            if "SafeRunSqlTool" in str(type(tool)) or "RunSqlTool" in str(type(tool)):
                                                sql_tool = tool
                                                break
                                    elif isinstance(tool_dict, list):
                                        for tool in tool_dict:
                                            if "SafeRunSqlTool" in str(type(tool)) or "RunSqlTool" in str(type(tool)):
                                                sql_tool = tool
                                                break
                                if sql_tool:
                                    break
                            
                            if sql_tool:
                                # Create a user for the tool execution
                                from vanna.core.user import User
                                user = User(
                                    id="api-chat-user",
                                    email="api-chat@local",
                                    group_memberships=["user"],
                                    metadata={"conversation_id": conversation_id},
                                )
                                
                                # Execute SQL using agent's tool
                                data = sql_tool.run(sql_query, user=user)
                                logger.info(
                                    "sql_executed_via_agent_tool",
                                    tool_type=type(sql_tool).__name__,
                                    row_count=len(data) if data else 0,
                                )
                            else:
                                # Fall back to manual execution if tool not found
                                logger.warning("sql_tool_not_found_falling_back_to_manual")
                                data, sql_error = execute_sql(sql_query)
                                if sql_error:
                                    ai_response += f"\n\n**SQL Execution Error:** {sql_error}"
                        except Exception as e:
                            logger.error("sql_execution_via_tool_failed", error=str(e))
                            sql_error = str(e)
                            ai_response += f"\n\n**SQL Execution Error:** {sql_error}"
                        
                        if not sql_error and data is not None:
                            row_count = len(data)
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
                                    f"Rows (up to {max_rows}): {json.dumps(preview_rows, ensure_ascii=False, cls=DatabaseJSONEncoder)}\n"
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
# Admin Memory Management Endpoints
# -----------------------------------------------------------------------------
def add_memory_admin_endpoints(app, agent: Agent):
    from fastapi import HTTPException
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field
    import uuid

    class TextMemoryItem(BaseModel):
        id: str
        content: str
        metadata: dict | None = None

    class ListTextMemoriesResponse(BaseModel):
        status: str
        items: list[TextMemoryItem]

    class CreateTextMemoryRequest(BaseModel):
        content: str = Field(..., min_length=1)

    class CreateGoldenQueryRequest(BaseModel):
        question: str = Field(..., min_length=1)
        sql: str = Field(..., min_length=1)

    def _require_http_chroma_memory() -> HttpChromaAgentMemory:
        mem = getattr(agent, "agent_memory", None)
        if not isinstance(mem, HttpChromaAgentMemory):
            raise HTTPException(
                status_code=500,
                detail="Agent memory is not HttpChromaAgentMemory; cannot manage raw Chroma collection.",
            )
        return mem

    @app.get("/api/admin/memory/text", response_model=ListTextMemoriesResponse, tags=["Training"])
    async def list_text_memories(limit: int = 100, offset: int = 0):
        try:
            mem = _require_http_chroma_memory()
            data = mem.raw_get(limit=limit, offset=offset)
            ids = data.get("ids") or []
            docs = data.get("documents") or []
            metas = data.get("metadatas") or []
            items = []
            for i in range(min(len(ids), len(docs))):
                items.append({"id": ids[i], "content": docs[i] or "", "metadata": (metas[i] if i < len(metas) else None)})
            return JSONResponse({"status": "success", "items": items})
        except HTTPException:
            raise
        except Exception as e:
            logger.error("admin_list_text_memories_failed", error=str(e))
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/admin/memory/text", tags=["Training"])
    async def create_text_memory(payload: CreateTextMemoryRequest):
        try:
            from vanna.core.tool import ToolContext
            from vanna.core.user import User

            admin_user = User(
                id="dashboard@admin",
                email="dashboard@admin",
                group_memberships=["admin"],
                metadata={},
            )
            context = ToolContext(
                user=admin_user,
                agent=agent,
                agent_memory=agent.agent_memory,
                conversation_id="dashboard-admin",
                request_id=str(uuid.uuid4()),
            )

            await agent.agent_memory.save_text_memory(content=payload.content, context=context)
            return JSONResponse({"status": "success"})
        except Exception as e:
            logger.error("admin_create_text_memory_failed", error=str(e))
            raise HTTPException(status_code=500, detail=str(e))

    @app.delete("/api/admin/memory/text/{memory_id}", tags=["Training"])
    async def delete_text_memory(memory_id: str):
        try:
            mem = _require_http_chroma_memory()
            mem.raw_delete(ids=[memory_id])
            return JSONResponse({"status": "success"})
        except HTTPException:
            raise
        except Exception as e:
            logger.error("admin_delete_text_memory_failed", error=str(e))
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/admin/golden_queries", response_model=ListTextMemoriesResponse, tags=["Training"])
    async def list_golden_queries(limit: int = 100, offset: int = 0):
        try:
            mem = _require_http_chroma_memory()
            data = mem.raw_get(
                limit=limit,
                offset=offset,
                where_document={"$contains": "GOLDEN_QUERY\n"},
            )
            ids = data.get("ids") or []
            docs = data.get("documents") or []
            metas = data.get("metadatas") or []
            items = []
            for i in range(min(len(ids), len(docs))):
                items.append({"id": ids[i], "content": docs[i] or "", "metadata": (metas[i] if i < len(metas) else None)})
            return JSONResponse({"status": "success", "items": items})
        except HTTPException:
            raise
        except Exception as e:
            logger.error("admin_list_golden_queries_failed", error=str(e))
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/admin/golden_queries", tags=["Training"])
    async def create_golden_query(payload: CreateGoldenQueryRequest):
        try:
            from vanna.core.tool import ToolContext
            from vanna.core.user import User

            admin_user = User(
                id="dashboard@admin",
                email="dashboard@admin",
                group_memberships=["admin"],
                metadata={},
            )
            context = ToolContext(
                user=admin_user,
                agent=agent,
                agent_memory=agent.agent_memory,
                conversation_id="dashboard-admin",
                request_id=str(uuid.uuid4()),
            )

            content = f"GOLDEN_QUERY\nQuestion: {payload.question}\nSQL: {payload.sql}"
            await agent.agent_memory.save_text_memory(content=content, context=context)
            return JSONResponse({"status": "success"})
        except Exception as e:
            logger.error("admin_create_golden_query_failed", error=str(e))
            raise HTTPException(status_code=500, detail=str(e))

    @app.delete("/api/admin/golden_queries/{memory_id}", tags=["Training"])
    async def delete_golden_query(memory_id: str):
        try:
            mem = _require_http_chroma_memory()
            mem.raw_delete(ids=[memory_id])
            return JSONResponse({"status": "success"})
        except HTTPException:
            raise
        except Exception as e:
            logger.error("admin_delete_golden_query_failed", error=str(e))
            raise HTTPException(status_code=500, detail=str(e))


# -----------------------------------------------------------------------------
# Streaming Chat Endpoint (SSE, UI-compatible)
# -----------------------------------------------------------------------------
def add_chat_sse_endpoint(app):
    """
    This function previously overrode VannaFastAPIServer's /api/vanna/v2/chat_sse endpoint.
    Now it does nothing, allowing VannaFastAPIServer to handle the SSE endpoint
    using the proper Agent -> Tool Registry -> RunSQLTool/SafeRunSQLTool + Agent Memory architecture.
    
    The VannaFastAPIServer provides a proper SSE endpoint that uses the agent architecture.
    """
    logger.info("VannaFastAPIServer will handle /api/vanna/v2/chat_sse endpoint with agent architecture")
    
    # Note: We're NOT removing VannaFastAPIServer's route anymore
    # The VannaFastAPIServer's endpoint will be used, which follows the proper architecture:
    # Agent -> Tool Registry -> RunSQLTool/SafeRunSQLTool + Agent Memory (Chroma)


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


# -----------------------------------------------------------------------------
# Debug Endpoints
# -----------------------------------------------------------------------------
def add_debug_endpoints(app, agent: Agent):
    """
    Add debug endpoints for troubleshooting.
    """
    from fastapi.responses import JSONResponse
    from fastapi import HTTPException
    
    @app.get("/debug/env")
    async def debug_env():
        """
        Debug endpoint to show environment configuration.
        """
        return JSONResponse({
            "OPENAI_MODEL": os.getenv("OPENAI_MODEL"),
            "MYSQL_DO_HOST": os.getenv("MYSQL_DO_HOST"),
            "MYSQL_DO_MAPA_DB": os.getenv("MYSQL_DO_MAPA_DB"),
            "MYSQL_DO_PORT": os.getenv("MYSQL_DO_PORT"),
            "VANNA_SIG_MAX_SKEW": os.getenv("VANNA_SIG_MAX_SKEW"),
            "AI_DEFAULT_LIMIT": os.getenv("AI_DEFAULT_LIMIT"),
            "AI_MAX_LIMIT": os.getenv("AI_MAX_LIMIT"),
            "AI_BLOCKED_RELATIONS": os.getenv("AI_BLOCKED_RELATIONS"),
            "AI_BLOCKED_COLS": os.getenv("AI_BLOCKED_COLS"),
            "CHROMA_PERSIST_DIR": os.getenv("CHROMA_PERSIST_DIR"),
            "CHROMA_COLLECTION": os.getenv("CHROMA_COLLECTION"),
            "CHROMA_SEED_VERSION": os.getenv("CHROMA_SEED_VERSION"),
        })
    
    @app.get("/debug/tools")
    async def debug_tools():
        """
        Debug endpoint to show registered tools.
        """
        tools = getattr(agent, "tool_registry", None)
        if not tools:
            return JSONResponse({"error": "No tool registry found"})
        
        out = {"tool_registry_type": str(type(tools))}
        for attr in ("_tools", "tools", "_registered_tools", "registered_tools", "local_tools"):
            if hasattr(tools, attr):
                try:
                    v = getattr(tools, attr)
                    out[attr] = list(v.keys()) if isinstance(v, dict) else [str(x) for x in v]
                except Exception as e:
                    out[attr] = f"ERR: {e}"
        return JSONResponse(out)
    
    @app.get("/debug/sql")
    async def debug_sql():
        """
        Debug endpoint to test SQL execution directly.
        """
        # Get SQL runner from agent
        tools = getattr(agent, "tool_registry", None)
        if not tools:
            raise HTTPException(status_code=500, detail="No tool registry found")
        
        # Find RunSqlTool
        sql_tool = None
        for attr in ("_tools", "tools", "_registered_tools", "registered_tools", "local_tools"):
            if hasattr(tools, attr):
                v = getattr(tools, attr)
                if isinstance(v, dict):
                    for tool_name, tool in v.items():
                        if "RunSqlTool" in str(type(tool)):
                            sql_tool = tool
                            break
                elif isinstance(v, list):
                    for tool in v:
                        if "RunSqlTool" in str(type(tool)):
                            sql_tool = tool
                            break
            if sql_tool:
                break
        
        if not sql_tool:
            raise HTTPException(status_code=500, detail="No RunSqlTool found")
        
        # Test SQL - use a simple generic query
        sql = """
        SELECT 1 as test_value, 'test' as test_string, NOW() as current_time
        """
        
        try:
            # Create a debug user
            from vanna.core.user import User
            user = User(id="debug", email="debug@local", group_memberships=["users"])
            
            # Run SQL
            rows = sql_tool.run(sql, user=user)
            return JSONResponse({"rows": rows})
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    main()
