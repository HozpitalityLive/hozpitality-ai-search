"""
Self-hosted Vanna 2.0 server with Gemini LLM.

Connects to PostgreSQL and BigQuery databases.
Run with: python main.py
"""

import os

from dotenv import load_dotenv
from fastapi import FastAPI
from vanna import AgentConfig

from vanna.core.agent import Agent
from vanna.core.registry import ToolRegistry
from vanna.core.user import User
from vanna.core.user.resolver import UserResolver
from vanna.core.user.request_context import RequestContext
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


def create_app() -> FastAPI:
    app = FastAPI(title="Vanna Text-to-SQL")

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
- Use PostgreSQL syntax.
- Always use LIMIT when returning rows.
- Only perform read-only SQL queries.
- After receiving the SQL result, summarize the result for the user.

Hozpitality Table Mapping:
- JOB / JOBS / JOB VACANCY / JOB VACANCIES / CAREER / OPENING queries MUST use `base_job`.
- Do NOT use a table named `jobs`.
- When the user asks to find, search, count, filter, or analyze jobs, use `base_job`.
- Before generating a complex job query, inspect the columns of `base_job` if the required column names are unknown.
- For job title searches, use the appropriate job_title column from `base_job`.
- For job location searches, use the appropriate job_city column from `base_job`.
- For job status searches, use the appropriate job_start_date column from `base_job`.
- For job expiry searches, use the appropriate job_end_date column from `base_job`.

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
        ),
        system_prompt_builder=system_prompt_builder,
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
