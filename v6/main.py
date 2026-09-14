"""
Hozpitality AI Search V6.1 — hybrid global search + Vanna text-to-SQL.

Connects to PostgreSQL and BigQuery databases.
Run with: python main.py
"""

import os
import re
from pathlib import Path
from typing import Optional

import psycopg2
import psycopg2.extras

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from vanna import AgentConfig
from vanna.core.agent.config import UiFeatures

from vanna.core.agent import Agent
from vanna.core.registry import ToolRegistry
from vanna.core.user import User
from vanna.core.user.resolver import UserResolver
from vanna.core.user.request_context import RequestContext
from vanna.core.workflow import WorkflowHandler, WorkflowResult
from vanna.components import UiComponent, RichTextComponent, SimpleTextComponent
from vanna.integrations.google.gemini import GeminiLlmService
from vanna.integrations.ollama.llm import OllamaLlmService
from vanna.integrations.chromadb.agent_memory import ChromaAgentMemory
from vanna.integrations.postgres.sql_runner import PostgresRunner
from vanna.integrations.bigquery.sql_runner import BigQueryRunner
from vanna.tools.run_sql import RunSqlTool
from vanna.tools.visualize_data import VisualizeDataTool
from vanna.core.system_prompt import DefaultSystemPromptBuilder
from vanna.servers.base import ChatHandler
from vanna.servers.fastapi.routes import register_chat_routes
from vanna.hozpitality.schema_intelligence import HozpitalitySchemaIntelligence
from vanna.hozpitality.global_search import GlobalSearchService

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


class LocalUserResolver(UserResolver):
    """Simple user resolver for personal/team use — returns a static admin user."""

    async def resolve_user(self, request_context: RequestContext) -> User:
        return User(
            id="local-user",
            username="admin",
            email="admin@localhost",
            group_memberships=["admin"],
        )


class HozpitalityWorkflowHandler(WorkflowHandler):
    """Deterministic front-door QA + fast searches for common Hozpitality intents.

    Common Hozpitality searches are executed directly against the authoritative
    source tables. This prevents the small local model from inventing columns,
    looping over failed SQL, or exposing SQL/tool protocol to the user.
    """

    GREETINGS = {
        "hi": "Hi! 👋 How can I help you with Hozpitality?",
        "hello": "Hello! 👋 What can I help you find on Hozpitality?",
        "hey": "Hey! 👋 What can I help you find?",
        "hi there": "Hi there! 👋 What would you like to find?",
        "hello there": "Hello there! 👋 What would you like to find?",
        "good morning": "Good morning! 👋 How can I help you?",
        "good afternoon": "Good afternoon! 👋 How can I help you?",
        "good evening": "Good evening! 👋 How can I help you?",
        "good day": "Good day! 👋 How can I help you?",
        "how are you": "I'm doing well and ready to help with Hozpitality.",
        "thanks": "You're welcome! 👋",
        "thank you": "You're welcome! 👋",
        "who are you": "I'm Hozpitality AI. I can help you find jobs, professionals, companies, products, events, articles and FAQs on Hozpitality.",
        "what can you do": "I can search Hozpitality for jobs, professionals, companies, marketplace products, events, articles and FAQs. Tell me what you're looking for.",
        "what does this platform do": "Hozpitality is a hospitality industry platform for jobs, professionals, companies, marketplace products, articles, events and more. I can help you search that information.",
        "what is this platform": "Hozpitality is a hospitality industry platform for jobs, professionals, companies, marketplace products, articles and events. I can search its data for you.",
        "what is hozpitality": "Hozpitality is a global hospitality industry platform connecting professionals, companies and hospitality businesses. I can search its jobs, articles, events, products and other data.",
        "help": "Try a request such as **Find waiter jobs in Dubai**, **Find the latest hospitality news**, or **Find hospitality events in UAE**.",
    }

    def __init__(self, pg_config: dict):
        self.pg_config = pg_config
        self.schema = HozpitalitySchemaIntelligence(self._connect)
        self.global_search = GlobalSearchService(pg_config, self.schema)

    def _connect(self):
        return psycopg2.connect(**self.pg_config)

    @staticmethod
    def _component(text: str) -> UiComponent:
        return UiComponent(
            rich_component=RichTextComponent(content=text, markdown=True),
            simple_component=SimpleTextComponent(text=text),
        )

    @staticmethod
    def _clean_excerpt(value, limit=220):
        if not value:
            return ""
        text = re.sub(r"\s+", " ", str(value)).strip()
        return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"

    @staticmethod
    def _article_terms(message: str):
        text = message.lower()
        # Remove intent/category words while preserving the actual topic.
        text = re.sub(r"\b(find|search|show|list|get|give|me|the|latest|recent|new|news|article|articles|story|stories|please|for|of|in|on|about|related|to)\b", " ", text)
        return [w for w in re.findall(r"[a-z0-9][a-z0-9&'_-]*", text) if len(w) > 2]

    async def try_handle(self, agent, user, conversation, message):
        normalized = re.sub(r"\s+", " ", message.strip().lower()).strip(" .!?'")

        if normalized in self.GREETINGS:
            return WorkflowResult(should_skip_llm=True, components=[self._component(self.GREETINGS[normalized])])

        if re.search(r"\bwhat can you do(?: for me)?\b", normalized) or re.search(r"\bwhat (?:does|is) (?:this )?(?:platform|hozpitality)\b", normalized):
            text = self.GREETINGS["what can you do"] if "what can you do" in normalized else self.GREETINGS["what does this platform do"]
            return WorkflowResult(should_skip_llm=True, components=[self._component(text)])

        # Fast global retrieval: search the master index first, resolve the
        # content_type/object_id pair, then enrich from the authoritative source
        # table. This handles natural-language searches across the entire site
        # without requiring a hard-coded handler for every Django model.
        dynamic_category = self.schema.resolve_category(normalized)
        if self._is_bare_article_request(normalized) and self._is_article_request(normalized):
            return WorkflowResult(
                should_skip_llm=True,
                components=[self._component("What topic or category should I search for in the articles? You can name any article topic or category and I’ll search it dynamically.")],
            )

        global_result = await self._global_search(normalized)
        if global_result:
            return WorkflowResult(should_skip_llm=True, components=[self._component(global_result)])

        # Article source-table fallback remains available if the index is stale
        # or an article-specific field is not represented in the index.
        if self._is_article_request(normalized) or dynamic_category is not None:
            if self._is_bare_article_request(normalized):
                return WorkflowResult(
                    should_skip_llm=True,
                    components=[self._component("What topic or category should I search for in the articles? For example: **latest hospitality news**, **Editor's Choice**, or **hotel openings**.")],
                )
            result = await self._search_articles(normalized)
            if result is not None:
                return WorkflowResult(should_skip_llm=True, components=[self._component(result)])

        clarification = self._clarification(normalized)
        if clarification:
            return WorkflowResult(should_skip_llm=True, components=[self._component(clarification)])

        return WorkflowResult(should_skip_llm=False)

    async def _global_search(self, message: str) -> Optional[str]:
        """Search the global master index and enrich hits from source tables."""
        try:
            # Aggregations/analytics should remain with the SQL agent; global
            # retrieval is intended for entity/document discovery.
            if re.search(r"\b(how many|count|average|sum|total|maximum|minimum|compare|trend|percentage)\b", message):
                return None
            return self.global_search.search(message, limit=10)
        except Exception as exc:
            print(f"Global search workflow failed: {exc}")
            return None

    @staticmethod
    def _is_article_request(message: str) -> bool:
        # "news" and "blog" are article aliases; actual categories are resolved
        # from base_category at runtime.
        return bool(
            re.search(
                r"\b(article|articles|news|blog|blogs|story|stories)\b",
                message,
            )
            or "article" in message
        )

    @classmethod
    def _is_bare_article_request(cls, message: str) -> bool:
        if re.search(r"\b(latest|recent|new)\s+(news|articles?|stories?)\b", message):
            return False
        return False if cls._article_terms(message) else True

    @staticmethod
    def _article_terms(message: str):
        text = message.lower()
        text = re.sub(
            r"\b(find|search|show|list|get|give|me|the|latest|recent|new|news|"
            r"article|articles|story|stories|please|for|of|in|on|about|related|to)\b",
            " ",
            text,
        )
        return [
            w for w in re.findall(r"[a-z0-9][a-z0-9&'_-]*", text)
            if len(w) > 2
        ]

    def _resolved_article_terms(self, message: str) -> list[str]:
        """Remove dynamically resolved taxonomy values from topic terms."""
        terms = self._article_terms(message)
        category = self.schema.resolve_category(message)
        country = self.schema.resolve_country(message)

        remove = set()
        for value in (category, country):
            if value:
                remove.update(
                    re.findall(r"[a-z0-9][a-z0-9&'_-]*", str(value.name).lower())
                )
        return [term for term in terms if term not in remove]

    async def _search_articles(self, message: str) -> Optional[str]:
        """Search articles with source-table + search-index fallback."""
        try:
            with self._connect() as conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    category = self.schema.resolve_category(message)
                    country = self.schema.resolve_country(message)
                    terms = self._resolved_article_terms(message)
                    category_name = category.name if category else None
                    category_like = f"%{category_name}%" if category_name else None

                    # Accept the normal Django values `publish` and `published`
                    # without assuming one exact spelling.
                    filters = ["LOWER(BTRIM(a.status)) LIKE 'publish%'"]
                    params = []

                    if category:
                        normalized_category = re.sub(r"[^a-z0-9]+", " ", category_name.lower()).strip()
                        filters.append("""(
                            a.category_id = %s
                            OR EXISTS (
                                SELECT 1
                                FROM master_search_mastersearchindex si_cat
                                JOIN django_content_type ct_cat ON ct_cat.id = si_cat.content_type_id
                                WHERE si_cat.object_id = a.id
                                  AND LOWER(ct_cat.model) = 'article'
                                  AND (
                                      si_cat.category_text ILIKE %s
                                      OR regexp_replace(LOWER(COALESCE(si_cat.category_text, '')), '[^a-z0-9]+', ' ', 'g') ILIKE %s
                                  )
                            )
                        )""")
                        params.extend([category.id, category_like, f"%{normalized_category}%"])

                    if country:
                        filters.append("""
                            EXISTS (
                                SELECT 1 FROM base_article_location al
                                WHERE al.article_id = a.id AND al.country_id = %s
                            )
                        """)
                        params.append(country.id)

                    if terms:
                        term_clauses = []
                        for term in terms[:8]:
                            like = f"%{term}%"
                            term_clauses.append("""(
                                a.title ILIKE %s OR a.sub_title ILIKE %s OR a.content ILIKE %s
                                OR EXISTS (
                                    SELECT 1
                                    FROM master_search_mastersearchindex si_term
                                    JOIN django_content_type ct_term ON ct_term.id = si_term.content_type_id
                                    WHERE si_term.object_id = a.id
                                      AND LOWER(ct_term.model) = 'article'
                                      AND (
                                          si_term.title ILIKE %s OR si_term.category_text ILIKE %s
                                          OR si_term.ai_keywords ILIKE %s OR si_term.content ILIKE %s
                                      )
                                )
                            )""")
                            params.extend([like, like, like, like, like, like, like])
                        filters.append("(" + " OR ".join(term_clauses) + ")")

                    cur.execute(f"""
                        SELECT DISTINCT a.id, a.title, a.sub_title, a.content, a.slug, a.created_at,
                               c.name AS category_name, co.name AS country_name,
                               a.thumbnail_url, a.youtube_link
                        FROM base_article a
                        LEFT JOIN base_category c ON c.id = a.category_id
                        LEFT JOIN LATERAL (
                            SELECT co1.name FROM base_article_location al1
                            JOIN countries co1 ON co1.id = al1.country_id
                            WHERE al1.article_id = a.id ORDER BY co1.name LIMIT 1
                        ) co ON TRUE
                        WHERE {' AND '.join(filters)}
                        ORDER BY a.created_at DESC NULLS LAST, a.id DESC
                        LIMIT 10
                    """, params)
                    rows = cur.fetchall()

                    # Controlled fallback: use only live, non-expired Article index
                    # rows if source status vocabulary prevents a match.
                    if not rows:
                        ff = [
                            "LOWER(ct.model) = 'article'",
                            "si.is_live = TRUE",
                            "(si.expires_at IS NULL OR si.expires_at >= CURRENT_TIMESTAMP)",
                        ]
                        fp = []
                        if category:
                            normalized_category = re.sub(r"[^a-z0-9]+", " ", category_name.lower()).strip()
                            ff.append("(si.category_text ILIKE %s OR regexp_replace(LOWER(COALESCE(si.category_text, '')), '[^a-z0-9]+', ' ', 'g') ILIKE %s)")
                            fp.extend([category_like, f"%{normalized_category}%"])
                        if country:
                            ff.append("si.location_text ILIKE %s")
                            fp.append(f"%{country.name}%")
                        if terms:
                            tc = []
                            for term in terms[:8]:
                                like = f"%{term}%"
                                tc.append("(si.title ILIKE %s OR si.category_text ILIKE %s OR si.ai_keywords ILIKE %s OR si.content ILIKE %s)")
                                fp.extend([like, like, like, like])
                            ff.append("(" + " OR ".join(tc) + ")")
                        cur.execute(f"""
                            SELECT DISTINCT a.id, a.title, a.sub_title, a.content, a.slug, a.created_at,
                                   c.name AS category_name, si.location_text AS country_name,
                                   a.thumbnail_url, a.youtube_link
                            FROM master_search_mastersearchindex si
                            JOIN django_content_type ct ON ct.id = si.content_type_id
                            JOIN base_article a ON a.id = si.object_id
                            LEFT JOIN base_category c ON c.id = a.category_id
                            WHERE {' AND '.join(ff)}
                            ORDER BY COALESCE(si.created_at, a.created_at) DESC NULLS LAST, a.id DESC
                            LIMIT 10
                        """, fp)
                        rows = cur.fetchall()
        except Exception as exc:
            print(f"Article direct search failed: {exc}")
            return None

        if not rows:
            return "I couldn't find any published articles matching that request. Try another topic, category, or location."

        descriptors = []
        if category:
            descriptors.append(f"category **{category.name}**")
        if country:
            descriptors.append(f"location **{country.name}**")
        if descriptors:
            intro = f"I found {len(rows)} published articles for " + " and ".join(descriptors) + "."
        elif terms:
            intro = f"I found {len(rows)} published articles matching your search."
        else:
            intro = f"Here are the {len(rows)} latest published articles."

        lines = [intro, ""]
        for idx, row in enumerate(rows, 1):
            title = str(row.get("title") or "Untitled article").strip()
            created = row.get("created_at")
            date_text = created.strftime("%Y-%m-%d") if hasattr(created, "strftime") else str(created or "").split(" ")[0]
            excerpt = self._clean_excerpt(row.get("sub_title") or row.get("content"))
            lines.append(f"**{idx}. {title}**")
            if row.get("category_name"): lines.append(f"- Category: {row['category_name']}")
            if row.get("country_name"): lines.append(f"- Location: {row['country_name']}")
            if date_text: lines.append(f"- Published: {date_text}")
            if excerpt: lines.append(f"- {excerpt}")
            if row.get("slug"): lines.append(f"- Slug: `{row['slug']}`")
            lines.append("")
        return "\n".join(lines).strip()

    @staticmethod
    def _clarification(message: str):
        if re.search(r"\b(find|search|show|list|get)\b", message) and re.search(r"\bjobs?\b", message):
            role_words = re.sub(r"\b(find|search|show|list|get|me|some|any|the|a|an|please|for|jobs?|job|vacancies?|openings?)\b", " ", message)
            if not re.sub(r"\s+", " ", role_words).strip():
                return "What type of job would you like me to find? You can also include a location, for example: **waiter jobs in Dubai**."

        if re.search(r"\b(find|search|show|list|get)\b", message) and re.search(r"\b(events?|event)\b", message):
            detail = re.sub(r"\b(find|search|show|list|get|me|some|any|the|a|an|please|for|events?|event)\b", " ", message)
            if not re.sub(r"\s+", "", detail):
                return "What type of event or location should I search for? For example: **hospitality events in Dubai**."

        if re.search(r"\b(find|search|show|list|get)\b", message) and re.search(r"\b(products?|marketplace)\b", message):
            detail = re.sub(r"\b(find|search|show|list|get|me|some|any|the|a|an|please|for|products?|product|marketplace)\b", " ", message)
            if not re.sub(r"\s+", "", detail):
                return "What product or category are you looking for? You can also include a location."
        return None

def create_app() -> FastAPI:
    app = FastAPI(title="Hozpitality AI Search V6", version="6.1.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "http://in.localhost:3000",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "service": "hozpitality-ai-v6",
            "version": "6.1.0",
        }

    # LLM
    provider = os.getenv("LLM_PROVIDER", "gemini").lower()
    if provider == "ollama":
        llm = OllamaLlmService(
            model=os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
            host=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
        )
    else:
        llm = GeminiLlmService(
            model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
            api_key=os.getenv("GOOGLE_API_KEY"),
        )

    # Agent memory (ChromaDB)
    memory = ChromaAgentMemory(
        persist_directory=str(BASE_DIR / "chroma_data"),
        collection_name="vanna_memory",
    )

    # Tools
    tools = ToolRegistry()

    # PostgreSQL
    pg_host = os.getenv("POSTGRES_HOST")
    
    if not pg_host:
        raise RuntimeError("POSTGRES_HOST is not configured")
    if pg_host:
        pg_runner = PostgresRunner(
            host=pg_host,
            port=int(os.getenv("POSTGRES_PORT", "5432")),
            database=os.getenv("POSTGRES_DATABASE"),
            user=os.getenv("POSTGRES_USER"),
            password=os.getenv("POSTGRES_PASSWORD"),
        )
        tools.register_local_tool(
            RunSqlTool(sql_runner=pg_runner),
            access_groups=[],
        )

    # BigQuery
    bq_project = os.getenv("BIGQUERY_PROJECT_ID")
    if bq_project:
        bq_cred_file = os.getenv("BIGQUERY_CREDENTIALS_FILE")
        bq_runner = BigQueryRunner(
            project_id=bq_project,
            cred_file_path=bq_cred_file,
        )
        tools.register_local_tool(
            RunSqlTool(
                sql_runner=bq_runner,
                custom_tool_name="run_bigquery_sql",
                custom_tool_description="Execute SQL queries against BigQuery",
            ),
            access_groups=[],
        )

    # Visualization
    tools.register_local_tool(VisualizeDataTool(), access_groups=[])

    # System prompt — tell the agent what database it's connected to
    db_name = os.getenv("POSTGRES_DATABASE", "unknown")
    schema_intelligence = HozpitalitySchemaIntelligence(lambda: psycopg2.connect(
        host=pg_host,
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        dbname=os.getenv("POSTGRES_DATABASE"),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
    ))
    schema_context = schema_intelligence.system_context()
    system_prompt_builder = DefaultSystemPromptBuilder(base_prompt=f"""You are Vanna, an AI data analyst assistant. Today's date is {__import__('datetime').date.today()}.

DATABASE: You are connected to a PostgreSQL database named '{db_name}'.
- Use PostgreSQL syntax for all SQL queries.
- To describe a table, use: SELECT column_name, data_type, is_nullable FROM information_schema.columns WHERE table_name = '<table>' ORDER BY ordinal_position
- Do NOT use PRAGMA, DESCRIBE, or SHOW commands — those are for other databases.
- Always use LIMIT instead of TOP for row limits.

Response Guidelines:
- You are connected to a live PostgreSQL database.
- For ANY question asking about data in the database, you MUST use the `run_sql` tool.
- NEVER ask the user to provide a table name if the database can be inspected to determine it.
- If you do not know which table contains the requested information, first use `run_sql` to inspect `information_schema.tables` and `information_schema.columns`.
- For counting records, generate and execute a SQL COUNT query.
- For searching records, generate and execute a SQL SELECT query.
- Do not answer database questions from general knowledge.
- Do not merely describe what SQL could be executed. Actually call `run_sql`.
- NEVER show SQL code to the user as the answer.
- NEVER ask the user for permission to run SQL. Execute the query immediately.
- Use PostgreSQL syntax.
- Always use LIMIT when returning rows.
- Only perform read-only SQL queries.
- After receiving the SQL result, summarize the result for the user.
- Default presentation is clean ChatGPT-style text. Do not use Markdown tables, CSV, raw JSON, SQL code blocks, or technical tool output unless the user explicitly asks for that format.
- When multiple records are returned, use a numbered or bulleted list. For jobs, make each job title a clickable Markdown link using the actual `job_link` value when present. Never invent a URL.
- Return up to 10 useful matching records for search requests unless the user asks for another amount.
- Never tell the user to approve or run a query. Execute `run_sql` internally.
- QA / CLARIFICATION: If a request is genuinely ambiguous or lacks a critical search parameter, ask one concise clarification question before calling `run_sql`. If the request is clear enough, search immediately without asking unnecessary questions.
- For job searches, prefer partial title matching with `job_title ILIKE '%term%'` rather than exact equality.
- For job searches, return useful fields such as `job_title`, `job_city`, `job_desc`, `job_status`, `job_start_date`, `job_end_date`, `job_link`, and `slug`; avoid `SELECT *` unless specifically requested.

Hozpitality Database Intelligence:
{schema_context}

Hozpitality Search Routing:
- master_search_mastersearchindex is the cross-module discovery index and contains indexed content for jobs, professionals, companies, marketplace products, events, and articles.
- Use master_search_mastersearchindex for broad Hozpitality searches where the content type is unclear or the user asks to search across the platform.
- For a clearly identified content type, prefer its source table so the response contains the authoritative fields and links.
- Jobs -> base_job
- Users/accounts -> user_accounts
- Professionals -> professionals
- Articles -> base_article
- Events -> base_event
- Marketplace products -> marketplace_product
- FAQs -> base_faq
- Never invent table names.
- If the user's wording is ambiguous enough that the correct content type or a critical search term cannot be determined, the QA layer will ask a clarification question before SQL is executed.
- For a job search, use job_title/job_city terms from the user. Do not require an exact title.
- For job availability, normally filter is_live = TRUE AND is_deleted = FALSE and exclude expired jobs when the user means currently available.
- For job results, prefer returning job_title, job_city, job_status, job_start_date, job_end_date, job_link, slug, and a short job_desc excerpt.
- When company information is requested for a job, LEFT JOIN user_accounts u ON u.id = j.company_id and use u.company_name or u.first_name/last_name only when those fields are appropriate.
- For broad cross-platform searches, query master_search_mastersearchindex first; use title, location_text, category_text, ai_keywords, user_name, content and slug.
- Treat master_search_mastersearchindex as retrieval/discovery, not the authoritative source when a source object can be resolved.
- Search ranking should favor exact title/category/person matches, then location/keywords/content/full-text relevance, then recency.
- Polymorphic tables use content_type_id + object_id. Resolve django_content_type.id before joining object_id to a source table; never assume object_id points to one fixed table.
- Article categories are dynamic records from base_category.name joined through base_article.category_id. Never hard-code a category such as Editor's Choice.
- Article locations are dynamic records from countries.name joined through base_article_location(article_id, country_id). Never hard-code country IDs.
- When the user names a category, country, or content type, resolve it against the database metadata before generating SQL.
- For source-specific searches, use the source table and its authoritative fields rather than relying only on the search index.

Hozpitality Table Mapping:
- JOB / JOBS / JOB VACANCY / JOB VACANCIES / CAREER / OPENING queries MUST use `base_job`.
- Do NOT use a table named `jobs`.
- When the user asks to find, search, count, filter, or analyze jobs, use `base_job`.
- Before generating a complex job query, inspect the columns of `base_job` if the required column names are unknown.
- For job title searches, use the appropriate job_title column from `base_job`.
- For job location searches, use the appropriate job_city column from `base_job`.
- For job status searches, use the `job_status` column from `base_job`.
- For job start-date searches, use the `job_start_date` column from `base_job`.
- For job expiry searches, use the `job_end_date` column from `base_job`.

Source-table guidance:
- base_job: authoritative job records and job links.
- user_accounts: account/company/supplier/student/guest records. Use user_type to distinguish account types when needed.
- professionals: professional-specific fields such as department, job_level, education_level, job_role, currently_working and current_company.
- base_article: article title, sub_title, content, category, location, created_at, youtube_link, thumbnail_url, slug, isFeatured, status, is_auto_renew_enabled.
- base_article DOES NOT have is_live, is_deleted, or publication_date. For latest articles use status = 'publish' ORDER BY created_at DESC.
- For Editor's Choice, use master_search_mastersearchindex.category_text (joined to base_article by object_id and Article content_type) rather than inventing an article field.
- base_event: event title, dates, city, country, type, status, website and details.
- marketplace_product: product title, type, condition, price, location, description, website_link, status and slug.
- base_faq: FAQ question and answer.
- master_search_mastersearchindex: cross-module indexed title, slug, location_text, category_text, ai_keywords, content, content_type/object_id, live/expiry metadata.

Job fields:
- Job title → `job_title`
- Job city → `job_city`
- Job description → `job_desc`
- Job status → `job_status`
- Job availability/live status → `is_live`
- Job start date → `job_start_date`
- Job expiry date → `job_end_date`
- Job creation date → `created_at`
- Job company → `company_id`
- Job URL → `job_link`
- Job slug → `slug`
- Featured job → `is_featured`
- Premium job → `is_premium`
- Deleted job → `is_deleted`

Job filtering rules:
- Normally exclude deleted jobs using `is_deleted = FALSE`.
- For currently available/live jobs, use `is_live = TRUE AND is_deleted = FALSE`.
- For expired jobs, use `job_end_date < CURRENT_DATE`.
- For active/non-expired jobs, consider `job_end_date IS NULL OR job_end_date >= CURRENT_DATE`, together with `is_deleted = FALSE`.

Examples:
- "Find waiter jobs" -> query `base_job`.
- "Find chef jobs in Dubai" -> query `base_job`.
- "How many jobs are available?" -> query `base_job`.
- "Show latest hotel jobs" -> query `base_job`.
- "Find housekeeping vacancies in UAE" -> query `base_job`.

Never invent a generic table such as `jobs` when a Hozpitality-specific table mapping exists.


"""

)

    # Agent
    # agent = Agent(
    #     llm_service=llm,
    #     tool_registry=tools,
    #     user_resolver=LocalUserResolver(),
    #     agent_memory=memory,
    #     system_prompt_builder=system_prompt_builder,
    # )

    workflow = HozpitalityWorkflowHandler({
        "host": pg_host,
        "port": int(os.getenv("POSTGRES_PORT", "5432")),
        "dbname": os.getenv("POSTGRES_DATABASE"),
        "user": os.getenv("POSTGRES_USER"),
        "password": os.getenv("POSTGRES_PASSWORD"),
    })

    agent = Agent(
        llm_service=llm,
        tool_registry=tools,
        user_resolver=LocalUserResolver(),
        agent_memory=memory,
        config=AgentConfig(
            stream_responses=False,
            temperature=0,
            max_tool_iterations=4,
            # Never expose internal tool names, arguments, or tool-call text
            # to end users. Tool execution remains enabled internally.
            ui_features=UiFeatures(feature_group_access={}),
        ),
        system_prompt_builder=system_prompt_builder,
        workflow_handler=workflow,
    )

    # Fast global-search API for the frontend and smoke tests. Analytics are
    # intentionally left to the Vanna SQL path.
    @app.get("/api/search")
    async def global_search_api(q: str, limit: int = 10):
        if not q or len(q.strip()) < 2:
            return {"query": q, "results": [], "count": 0}
        results = workflow.global_search.search_hits(q, max(1, min(limit, 20)))
        return {
            "query": q,
            "count": len(results),
            "results": results,
            "stats": workflow.global_search.last_stats,
        }

    @app.get("/api/search/health")
    async def global_search_health():
        # Read-only checks; no full-table scan.
        try:
            with workflow._connect() as conn, conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM public.master_search_mastersearchindex")
                total = cur.fetchone()[0]
                cur.execute("SELECT COUNT(*) FROM public.master_search_mastersearchindex WHERE search_vector_v6 IS NOT NULL")
                fts_ready = cur.fetchone()[0]
                cur.execute("SELECT COUNT(*) FROM public.master_search_mastersearchindex WHERE embedding IS NOT NULL")
                vectors_ready = cur.fetchone()[0]
                cur.execute("""SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname='vector'),
                                    EXISTS (SELECT 1 FROM pg_extension WHERE extname='pg_trgm'),
                                    EXISTS (SELECT 1 FROM pg_extension WHERE extname='unaccent')""")
                vector_ext, trgm_ext, unaccent_ext = cur.fetchone()
                return {
                    "status": "ok",
                    "master_index_rows": total,
                    "fts_ready_rows": fts_ready,
                    "vector_rows": vectors_ready,
                    "extensions": {"vector": vector_ext, "pg_trgm": trgm_ext, "unaccent": unaccent_ext},
                }
        except Exception as exc:
            return {"status": "degraded", "error": str(exc)}

    @app.get("/api/search/metrics")
    async def global_search_metrics():
        return {"stats": workflow.global_search.last_stats}

    # Schema explorer endpoint
    @app.get("/api/schema")
    async def get_schema():
        """Return database schema tree for the sidebar explorer."""
        import psycopg2
        import psycopg2.extras

        pg_host = os.getenv("POSTGRES_HOST")
        if not pg_host:
            return {"schemas": []}

        conn = psycopg2.connect(
            host=pg_host,
            port=os.getenv("POSTGRES_PORT", "5432"),
            dbname=os.getenv("POSTGRES_DATABASE"),
            user=os.getenv("POSTGRES_USER"),
            password=os.getenv("POSTGRES_PASSWORD"),
        )
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("""
            SELECT table_schema, table_name, column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_schema NOT IN ('pg_catalog', 'information_schema')
            ORDER BY table_schema, table_name, ordinal_position
        """)
        rows = cur.fetchall()
        cur.close()
        conn.close()

        # Build tree: schema > table > columns
        schemas: dict = {}
        for row in rows:
            s = schemas.setdefault(row["table_schema"], {})
            t = s.setdefault(row["table_name"], [])
            t.append({
                "name": row["column_name"],
                "type": row["data_type"],
                "nullable": row["is_nullable"] == "YES",
            })

        result = []
        for schema_name, tables in sorted(schemas.items()):
            result.append({
                "name": schema_name,
                "tables": [
                    {"name": tname, "columns": cols}
                    for tname, cols in sorted(tables.items())
                ],
            })

        return {"schemas": result}

    # Serve local web component build
    chat_handler = ChatHandler(agent=agent)
    register_chat_routes(app, chat_handler)

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8085"))
    print(f"Starting Vanna server at http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)
