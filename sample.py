# app.py - Vanna 2.x (tested target: vanna==2.0.1)
#
# Features:
# - Signed-header auth via HMAC (PHP proxy compatible)
# - Nonce replay protection + timestamp skew
# - Safe SQL tool wrapper (SELECT-only, no SELECT *, LIMIT enforcement, optional deny lists)
# - MySQLRunner + ChromaAgentMemory (seeded w/ schema + golden queries, de-duplicated via marker file)
# - Vanna FastAPI server routes incl. SSE chat
# - Debug endpoints: /health, /debug/env, /debug/tools, /debug/sql (direct tool call)
#
# Environment (.env recommended at /opt/vanna/.env)
#
# Docker install (example):
#   pip install --no-cache-dir -U "vanna[fastapi,openai,mysql,chromadb]==2.0.1"

import os
import time
import hmac
import hashlib
import logging
import re
from typing import Dict, List, Set, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from vanna import Agent, AgentConfig
from vanna.core.registry import ToolRegistry
from vanna.integrations.openai import OpenAILlmService
from vanna.integrations.mysql import MySQLRunner
from vanna.integrations.chromadb import ChromaAgentMemory
from vanna.tools import RunSqlTool

from vanna.core.user import UserResolver, User, RequestContext
from vanna.servers.fastapi import VannaFastAPIServer


# ------------------------------------------------------------
# Bootstrap logging + env
# ------------------------------------------------------------
load_dotenv("/opt/vanna/.env", override=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app")

print("HOTRELOAD CHECK", __file__)
logger.info("APP BUILD CODENAME monkey")
logger.info("RELOAD TIME %s", time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))
logger.info("ENV_VERSION: %s", os.getenv("ENV_VERSION"))
logger.info("AI_MODEL: %s", os.getenv("OPENAI_MODEL"))
logger.info("DOTENV CHECK: %s %s", os.getenv("OPENAI_MODEL"), os.getenv("MYSQL_DO_HOST"))


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def _csv_to_set(value: Optional[str]) -> Set[str]:
    if not value:
        return set()
    return {v.strip().lower() for v in value.split(",") if v.strip()}


# ------------------------------------------------------------
# Signed-header resolver (RequestContext-based)
# ------------------------------------------------------------
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


# ------------------------------------------------------------
# Safe SQL tool wrapper (Vanna 2.x expects RunSqlTool.run)
# ------------------------------------------------------------
class SafeRunSqlTool(RunSqlTool):
    """
    Defense-in-depth. Combineer dit met:
    - DB user die enkel SELECT heeft op de AI views
    - Views die alleen toegelaten kolommen bevatten
    """

    DEFAULT_LIMIT = int(os.getenv("AI_DEFAULT_LIMIT", "200"))
    MAX_LIMIT = int(os.getenv("AI_MAX_LIMIT", "500"))

    # Optional deny lists from env
    BLOCKED_RELATIONS = _csv_to_set(os.getenv("AI_BLOCKED_RELATIONS"))
    BLOCKED_COLS = _csv_to_set(os.getenv("AI_BLOCKED_COLS"))

    def run(self, sql: str, user=None):
        logger.warning("SAFE run() HIT user=%r sql=%r", getattr(user, "id", None), (sql or "")[:200])

        s = (sql or "").strip()
        s_low = s.lower()

        # single statement only
        if ";" in s_low.rstrip(";"):
            raise ValueError("Multiple statements are not allowed.")

        # only SELECT/WITH
        if not (s_low.startswith("select") or s_low.startswith("with")):
            raise ValueError("Only SELECT/WITH queries are allowed.")

        # enforce only our AI view (strongly recommended)
        if "vw_aankoopbonlijnen_ai" not in s_low:
            raise ValueError("Query must use vw_aankoopbonlijnen_ai.")

        # no SELECT *
        if re.search(r"\bselect\s+\*\b", s_low):
            raise ValueError("SELECT * is not allowed.")

        # blocked relations (optional)
        for rel in self.BLOCKED_RELATIONS:
            if re.search(rf"\b{re.escape(rel)}\b", s_low):
                raise ValueError(f"Relation '{rel}' is not allowed.")

        # blocked columns (optional)
        for col in self.BLOCKED_COLS:
            if re.search(rf"\b{re.escape(col)}\b", s_low):
                raise ValueError(f"Column '{col}' is not allowed.")

        # enforce LIMIT (and cap)
        m = re.search(r"\blimit\s+(\d+)\b", s_low)
        if not m:
            s = s.rstrip() + f" LIMIT {self.DEFAULT_LIMIT}"
        else:
            lim = int(m.group(1))
            if lim > self.MAX_LIMIT:
                s = re.sub(r"(?i)\blimit\s+\d+\b", f"LIMIT {self.MAX_LIMIT}", s, count=1)

        return super().run(s, user=user)


# ------------------------------------------------------------
# Memory seeding
# ------------------------------------------------------------
SCHEMA_DOC = """
Database context (MySQL) - AI query scope:

Toegestane bron (enige bron):
- vw_aankoopbonlijnen_ai

Kolommen (belangrijkste):
- datum_bestelling (DATE): bestel-/boekingdatum (primaire tijdfilter)
- l_naam (VARCHAR): leveranciernaam (primaire leverancierfilter)
- l_nr (INT): leveranciersnummer
- artikelcode, productcode
- omschrijving (TEXT)
- merk, groep, subgroep
- hoev, prijs, tot_prijs
- klantnaam, project, project_omschrijving, bestemming

Business regels:
- Leveranciernaam staat in l_naam (l_nr is nummer).
- ref_leverancier is NIET de primaire leverancierfilter.
- 'Thermokey' kan voorkomen in l_naam, merk of omschrijving.
- Tekstfilters: UPPER(TRIM(col)) LIKE '%TERM%'.
- De view is lijnniveau: elke rij = 1 aankoopbonlijn (item). Eén aankoopbon heeft meerdere lijnen.
- Als de gebruiker "bonnen/aankoopbonnen" vraagt: geef bonniveau (GROUP BY aankoopbon).
- Als de gebruiker "lijnen/items/artikelen" vraagt: geef lijnniveau (WHERE aankoopbon = ...).
- Voor aantal bonnen: COUNT(DISTINCT aankoopbon). Voor bon totaal: SUM(tot_prijs) GROUP BY aankoopbon.

Query regels:
- Alleen SELECT/WITH, geen SELECT *, altijd LIMIT.
- Gebruik duidelijke aliases.
- Datums zijn DATE (YYYY-MM-DD).
- Default periode: laatste 90 dagen als geen periode gegeven.
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
BONNIVEAU - laatste bonnen:
SELECT
  aankoopbon,
  MAX(datum_bestelling) AS datum_bestelling,
  MAX(l_naam) AS leverancier,
  COUNT(*) AS aantal_lijnen,
  ROUND(SUM(tot_prijs), 2) AS bon_totaal
FROM vw_aankoopbonlijnen_ai
GROUP BY aankoopbon
ORDER BY datum_bestelling DESC
LIMIT 50;
""".strip(),
        {"type": "example_query", "topic": "bonniveau_recent"},
    ),
    (
        """
LIJNNIVEAU - details van één bon (vervang 123456 door het gewenste bonnummer):
SELECT
  aankoopbon, datum_bestelling, l_naam, artikelcode, omschrijving, hoev, prijs, tot_prijs
FROM vw_aankoopbonlijnen_ai
WHERE aankoopbon = 123456
ORDER BY id ASC
LIMIT 200;
""".strip(),
        {"type": "example_query", "topic": "bon_detail"},
    ),
    (
        """
THERMOKEY - laatste 90 dagen:
SELECT datum_bestelling, l_naam, merk, artikelcode, LEFT(omschrijving, 120) AS omschrijving, tot_prijs
FROM vw_aankoopbonlijnen_ai
WHERE datum_bestelling >= DATE_SUB(CURDATE(), INTERVAL 90 DAY)
  AND (
    UPPER(TRIM(l_naam)) LIKE '%THERMOKEY%'
    OR UPPER(TRIM(merk)) LIKE '%THERMOKEY%'
    OR UPPER(TRIM(omschrijving)) LIKE '%THERMOKEY%'
  )
ORDER BY datum_bestelling DESC
LIMIT 200;
""".strip(),
        {"type": "example_query", "topic": "thermokey"},
    ),
    (
        """
TOP leveranciers op spend (12m):
SELECT l_naam, COUNT(DISTINCT aankoopbon) AS aantal_bonnen, ROUND(SUM(tot_prijs),2) AS totaal
FROM vw_aankoopbonlijnen_ai
WHERE datum_bestelling >= DATE_SUB(CURDATE(), INTERVAL 12 MONTH)
GROUP BY l_naam
ORDER BY totaal DESC
LIMIT 50;
""".strip(),
        {"type": "example_query", "topic": "top_suppliers"},
    ),
]


# ------------------------------------------------------------
# App factory
# ------------------------------------------------------------
def build_app() -> FastAPI:
    # MySQL runner
    sql_runner = MySQLRunner(
        host=os.environ["MYSQL_DO_HOST"],
        port=int(os.getenv("MYSQL_DO_PORT", "25060")),
        database=os.environ["MYSQL_DO_MAPA_DB"],
        user=os.environ["MYSQL_DO_USER"],
        password=os.environ["MYSQL_DO_PASSWORD"],
        ssl_ca=os.getenv("MYSQL_DO_SSL_CA"),
    )

    # Tool memory (Chroma)
    persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_memory")
    memory = ChromaAgentMemory(
        persist_directory=persist_dir,
        collection_name=os.getenv("CHROMA_COLLECTION", "tool_memories"),
    )

    # Seed memory only once per version (avoid duplicates on restart)
    seed_version = int(os.getenv("CHROMA_SEED_VERSION", "1"))
    marker_path = os.path.join(persist_dir, f".seeded_v{seed_version}")

    if not os.path.exists(marker_path):
        seed_memory(
            memory,
            SCHEMA_DOC,
            {"type": "db_schema", "scope": "vw_aankoopbonlijnen_ai", "version": seed_version},
        )
        for text, meta in GOLDEN_QUERIES:
            seed_memory(
                memory,
                text.strip(),
                {"scope": "vw_aankoopbonlijnen_ai", "version": seed_version, **meta},
            )
        try:
            os.makedirs(persist_dir, exist_ok=True)
            with open(marker_path, "w", encoding="utf-8") as f:
                f.write(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))
        except Exception:
            pass

    # Tools
    tools = ToolRegistry()
    tools.register_local_tool(
        SafeRunSqlTool(sql_runner=sql_runner),
        access_groups=["users", "user", "admin"],
    )

    # LLM
    llm = OpenAILlmService(
        api_key=os.getenv("OPENAI_API_KEY"),
        model=(os.getenv("OPENAI_MODEL") or "gpt-4o").strip(),
    )

    # Resolver
    resolver = SignedHeaderUserResolver(
        secret=os.getenv("VANNA_PROXY_HMAC_SECRET", ""),
        max_skew_seconds=int(os.getenv("VANNA_SIG_MAX_SKEW", "3600")),
        nonce_ttl_seconds=int(os.getenv("VANNA_NONCE_TTL", "900")),
        max_nonce_cache=int(os.getenv("VANNA_NONCE_CACHE", "10000")),
    )

    # Agent config: reinforce tool usage and output-first behavior
    cfg = AgentConfig()
    force_txt = """
Je hebt een SQL tool genaamd run_sql die echte MySQL SELECT/WITH queries uitvoert.
Gebruik uitsluitend de view `vw_aankoopbonlijnen_ai` (nooit raw tables).

WERKWIJZE (output-first):
1) Als de vraag database-gerelateerd is, voer onmiddellijk een eerste SELECT uit via run_sql.
2) Als de gebruiker geen periode geeft: gebruik standaard de laatste 90 dagen op datum_bestelling.
3) Toon altijd resultaten (ook als 0 rijen: zeg dat het 0 is en geef 1 suggestie om te verfijnen).
4) Stel maximaal 1 verduidelijkingsvraag, en alleen als het echt niet eenduidig is.

SQL REGELS:
- Alleen SELECT of WITH (geen mutaties).
- Nooit SELECT *; kies expliciete kolommen.
- Voeg altijd een LIMIT toe (max 200 tenzij gevraagd).
- Bij tekstfilters: gebruik robuust UPPER(TRIM(col)) LIKE '%TERM%'.
- Leverancierfilter: gebruik `l_naam` (niet ref_leverancier).
- 'Thermokey' kan voorkomen in l_naam, merk of omschrijving; controleer in die volgorde.

BONNIVEAU vs LIJNNIVEAU:
- vw_aankoopbonlijnen_ai is lijnniveau (1 rij = 1 bonlijn/item). Eén aankoopbon heeft meerdere lijnen.
- Als gebruiker "bonnen/aankoopbonnen" vraagt: toon bonniveau (GROUP BY aankoopbon, SUM(tot_prijs)).
- Als gebruiker "lijnen/items/artikelen" vraagt: toon lijnniveau (WHERE aankoopbon = ...).
- Voor aantal bonnen: COUNT(DISTINCT aankoopbon).

OUTPUT:
- Geef een korte toelichting + tabelresultaat.
""".strip()

    for attr in ("system_prompt", "instructions", "prompt", "agent_instructions"):
        if hasattr(cfg, attr):
            try:
                setattr(cfg, attr, force_txt)
            except Exception:
                pass

    # Agent
    agent = Agent(
        llm_service=llm,
        tool_registry=tools,
        user_resolver=resolver,
        agent_memory=memory,
        config=cfg,
    )

    # Create FastAPI app via Vanna server wrapper
    app = VannaFastAPIServer(agent).create_app()

    # Debug middleware: log incoming headers on chat endpoints
    @app.middleware("http")
    async def debug_incoming_headers(request: Request, call_next):
        if request.url.path.endswith("/chat_sse") or request.url.path.endswith("/chat"):
            try:
                logger.error("RAW FASTAPI HEADERS (%s): %r", request.url.path, dict(request.headers))
            except Exception:
                pass
        return await call_next(request)

    # Health
    @app.get("/health")
    async def health():
        return {"ok": True}

    # Debug env
    @app.get("/debug/env")
    async def debug_env():
        return {
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
        }

    # Debug tools
    @app.get("/debug/tools")
    async def debug_tools():
        out = {"tool_registry_type": str(type(tools))}
        for attr in ("_tools", "tools", "_registered_tools", "registered_tools", "local_tools"):
            if hasattr(tools, attr):
                try:
                    v = getattr(tools, attr)
                    out[attr] = list(v.keys()) if isinstance(v, dict) else [str(x) for x in v]
                except Exception as e:
                    out[attr] = f"ERR: {e}"
        return out

    # Direct SQL tool test (bypasses LLM)
    @app.get("/debug/sql")
    async def debug_sql():
        sql = """
        SELECT datum_bestelling, l_naam, artikelcode, omschrijving, hoev, tot_prijs
        FROM vw_aankoopbonlijnen_ai
        ORDER BY datum_bestelling DESC
        LIMIT 3
        """
        tool = SafeRunSqlTool(sql_runner=sql_runner)
        u = User(id="debug", email="debug@local", group_memberships=["users"])
        rows = tool.run(sql, user=u)
        return {"rows": rows}

    # Error mapping (helps debugging auth/tool policy errors)
    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError):
        return JSONResponse(status_code=400, content={"error": str(exc)})

    return app


app = build_app()
