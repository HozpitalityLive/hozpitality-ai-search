"""
Self-hosted Vanna 2.0 server with Gemini LLM.

Connects to PostgreSQL and BigQuery databases.
Run with: python main.py
"""

import os
import re

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

load_dotenv()


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
    """Fast, deterministic front-door QA layer for Hozpitality chat.

    Handles greetings locally and asks for clarification only when a request
    lacks a critical parameter. Clear database questions continue to Vanna.
    """

    GREETINGS = {
        "hi": "Hi! 👋 How can I help you with Hozpitality?",
        "hello": "Hello! 👋 What can I help you find on Hozpitality?",
        "hey": "Hey! 👋 How can I help you today?",
        "hi there": "Hi there! 👋 What would you like to find?",
        "hello there": "Hello! 👋 What would you like to find?",
        "good morning": "Good morning! 👋 How can I help you?",
        "good afternoon": "Good afternoon! 👋 How can I help you?",
        "good evening": "Good evening! 👋 How can I help you?",
        "good day": "Good day! 👋 How can I help you?",
        "how are you": "I'm doing well and ready to help with Hozpitality.",
        "thanks": "You're welcome! 👋",
        "thank you": "You're welcome! 👋",
        "who are you": "I'm Hozpitality AI. I can search Hozpitality jobs, professionals, companies, marketplace products, events, articles and FAQs.",
        "what can you do": "I can search Hozpitality data and help you find jobs, professionals, companies, products, events, articles and FAQs.",
        "help": "I can search Hozpitality data for you. Try: “Find waiter jobs in Dubai” or “Find hospitality events in UAE”.",
    }

    async def try_handle(self, agent, user, conversation, message):
        normalized = re.sub(r"\s+", " ", message.strip().lower()).strip(" .!?")
        if normalized in self.GREETINGS:
            text = self.GREETINGS[normalized]
            component = UiComponent(
                rich_component=RichTextComponent(content=text, markdown=True),
                simple_component=SimpleTextComponent(text=text),
            )
            return WorkflowResult(should_skip_llm=True, components=[component])

        # Critical ambiguity gates. Do not slow down clear questions.
        clarification = self._clarification(normalized)
        if clarification:
            component = UiComponent(
                rich_component=RichTextComponent(content=clarification, markdown=True),
                simple_component=SimpleTextComponent(text=clarification),
            )
            return WorkflowResult(should_skip_llm=True, components=[component])

        return WorkflowResult(should_skip_llm=False)

    @staticmethod
    def _clarification(message: str):
        # A generic "find/search/show jobs" request does not identify a useful
        # role or search term. Ask one question instead of making a broad query.
        if re.search(r"\b(find|search|show|list|get)\b", message) and re.search(r"\bjobs?\b", message):
            role_words = re.sub(r"\b(find|search|show|list|get|me|some|any|the|a|an|please|for|jobs?|job|vacancies?|openings?)\b", " ", message)
            role_words = re.sub(r"\s+", " ", role_words).strip()
            if not role_words:
                return "What type of job would you like me to find? You can also include a location, for example: **waiter jobs in Dubai**."

        if re.search(r"\b(find|search|show|list|get)\b", message) and re.search(r"\b(events?|event)\b", message):
            detail = re.sub(r"\b(find|search|show|list|get|me|some|any|the|a|an|please|for|events?|event)\b", " ", message)
            if not re.sub(r"\s+", "", detail):
                return "What type of event or location should I search for? For example: **hospitality events in Dubai**."

        if re.search(r"\b(find|search|show|list|get)\b", message) and re.search(r"\b(products?|marketplace)\b", message):
            detail = re.sub(r"\b(find|search|show|list|get|me|some|any|the|a|an|please|for|products?|product|marketplace)\b", " ", message)
            if not re.sub(r"\s+", "", detail):
                return "What product or category are you looking for? You can also include a location."

        if re.search(r"\b(find|search|show|list|get)\b", message) and re.search(r"\b(articles?|news)\b", message):
            detail = re.sub(r"\b(find|search|show|list|get|me|some|any|the|a|an|please|for|articles?|article|news)\b", " ", message)
            if not re.sub(r"\s+", "", detail):
                return "What topic or category should I search for in the articles?"

        return None

def create_app() -> FastAPI:
    app = FastAPI(title="Vanna Text-to-SQL")

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
            "service": "hozpitality-ai-v5",
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
        persist_directory="./chroma_data",
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
- For broad cross-platform searches, query master_search_mastersearchindex first; use its indexed title, location_text, category_text, ai_keywords and content to identify matching content.
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
- base_article: article title, content, category, status, slug and publication data.
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

    agent = Agent(
        llm_service=llm,
        tool_registry=tools,
        user_resolver=LocalUserResolver(),
        agent_memory=memory,
        config=AgentConfig(
            stream_responses=False,
            temperature=0,
            # Never expose internal tool names, arguments, or tool-call text
            # to end users. Tool execution remains enabled internally.
            ui_features=UiFeatures(feature_group_access={}),
        ),
        system_prompt_builder=system_prompt_builder,
        workflow_handler=HozpitalityWorkflowHandler(),
    )

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
    port = int(os.getenv("PORT", "8084"))
    print(f"Starting Vanna server at http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)
