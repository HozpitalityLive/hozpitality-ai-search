"""Generic SQL query execution tool with dependency injection."""

from typing import Any, Dict, List, Optional, Type, cast
import math
import uuid
from vanna.core.tool import Tool, ToolContext, ToolResult
from vanna.components import (
    UiComponent,
    DataFrameComponent,
    NotificationComponent,
    ComponentType,
    SimpleTextComponent,
)
from vanna.capabilities.sql_runner import SqlRunner, RunSqlToolArgs
from vanna.capabilities.file_system import FileSystem
from vanna.integrations.local import LocalFileSystem


class RunSqlTool(Tool[RunSqlToolArgs]):
    """Tool that executes SQL queries using an injected SqlRunner implementation."""

    def __init__(
        self,
        sql_runner: SqlRunner,
        file_system: Optional[FileSystem] = None,
        custom_tool_name: Optional[str] = None,
        custom_tool_description: Optional[str] = None,
    ):
        """Initialize the tool with a SqlRunner implementation.

        Args:
            sql_runner: SqlRunner implementation that handles actual query execution
            file_system: FileSystem implementation for saving results (defaults to LocalFileSystem)
            custom_tool_name: Optional custom name for the tool (overrides default "run_sql")
            custom_tool_description: Optional custom description for the tool (overrides default description)
        """
        self.sql_runner = sql_runner
        self.file_system = file_system or LocalFileSystem()
        self._custom_name = custom_tool_name
        self._custom_description = custom_tool_description

    @property
    def name(self) -> str:
        return self._custom_name if self._custom_name else "run_sql"

    @property
    def description(self) -> str:
        return (
            self._custom_description
            if self._custom_description
            else "Execute SQL queries against the configured database"
        )

    def get_args_schema(self) -> Type[RunSqlToolArgs]:
        return RunSqlToolArgs

    @staticmethod
    def _markdown_table(records: List[Dict[str, Any]], columns: List[str], max_rows: int = 10) -> str:
        """Build a compact user-facing markdown table from query results."""
        if not records or not columns:
            return "No matching records found."

        # Prefer useful columns and keep the chat readable.
        preferred = [
            "job_title", "job_city", "job_status", "job_start_date",
            "job_end_date", "job_link", "slug"
        ]
        selected = [c for c in preferred if c in columns]
        if not selected:
            selected = columns[:6]
        selected = selected[:6]

        def cell(value: Any) -> str:
            if value is None:
                return "—"
            text = str(value).replace("|", "\\|").replace("\\n", " ").replace("\\r", " ")
            if len(text) > 90:
                text = text[:87] + "..."
            return text

        lines = [
            "| " + " | ".join(selected) + " |",
            "| " + " | ".join(["---"] * len(selected)) + " |",
        ]
        for record in records[:max_rows]:
            lines.append("| " + " | ".join(cell(record.get(c)) for c in selected) + " |")

        return "\\n".join(lines)

    async def execute(self, context: ToolContext, args: RunSqlToolArgs) -> ToolResult:
        """Execute a SQL query using the injected SqlRunner."""
        try:
            # Use the injected SqlRunner to execute the query
            df = await self.sql_runner.run_sql(args, context)

            # Determine query type
            query_type = args.sql.strip().upper().split()[0]

            if query_type == "SELECT":
                # Handle SELECT queries with results
                if df.empty:
                    result = "Query executed successfully. No rows returned."
                    ui_component = UiComponent(
                        rich_component=DataFrameComponent(
                            rows=[],
                            columns=[],
                            title="Query Results",
                            description="No rows returned",
                        ),
                        simple_component=SimpleTextComponent(text=result),
                    )
                    metadata = {
                        "row_count": 0,
                        "columns": [],
                        "query_type": query_type,
                        "results": [],
                    }
                else:
                    # Convert DataFrame to records with JSON-safe types
                    # pandas/numpy types (int64, float64, NaN, NaT) cause
                    # Pydantic serialization errors — convert to native Python
                    import pandas as pd
                    import numpy as np
                    import datetime
                    from decimal import Decimal

                    def _sanitize_value(v: Any) -> Any:
                        if v is None:
                            return None
                        if isinstance(v, type(pd.NaT)):
                            return None
                        if isinstance(v, float) and math.isnan(v):
                            return None
                        if isinstance(v, (np.integer,)):
                            return int(v)
                        if isinstance(v, (np.floating,)):
                            return float(v)
                        if isinstance(v, np.bool_):
                            return bool(v)
                        if isinstance(v, Decimal):
                            return float(v)
                        if isinstance(v, (pd.Timestamp, np.datetime64)):
                            return str(v)
                        if isinstance(v, (datetime.date, datetime.datetime, datetime.time)):
                            return str(v)
                        if isinstance(v, datetime.timedelta):
                            return str(v)
                        if isinstance(v, bytes):
                            return v.decode("utf-8", errors="replace")
                        if isinstance(v, (str, int, float, bool)):
                            return v
                        # Fallback: convert anything else to string
                        return str(v)

                    results_data = [
                        {k: _sanitize_value(v) for k, v in row.items()}
                        for row in df.to_dict("records")
                    ]
                    columns = df.columns.tolist()
                    row_count = len(df)

                    # Write DataFrame to CSV file for downstream tools
                    file_id = str(uuid.uuid4())[:8]
                    filename = f"query_results_{file_id}.csv"
                    csv_content = df.to_csv(index=False)
                    await self.file_system.write_file(
                        filename, csv_content, context, overwrite=True
                    )

                    # Send compact, labeled records to the LLM. Never send CSV/table
                    # presentation instructions as the user-facing result.
                    def _display(v: Any, limit: int = 500) -> str:
                        if v is None:
                            return "—"
                        text = str(v).replace("\n", " ").replace("\r", " ").strip()
                        return text if len(text) <= limit else text[: limit - 1] + "…"

                    result_lines = [
                        f"DATABASE RESULTS: {row_count} row(s).",
                        "USER PRESENTATION RULE: Return normal ChatGPT-style prose/list items. Do not output SQL, JSON, CSV, or Markdown tables unless the user explicitly asks for a table.",
                    ]
                    for idx, record in enumerate(results_data[:25], start=1):
                        parts = []
                        for col in columns:
                            value = record.get(col)
                            if value is None or value == "":
                                continue
                            parts.append(f"{col}={_display(value)}")
                        result_lines.append(f"{idx}. " + "; ".join(parts))
                    if row_count > 25:
                        result_lines.append(f"Additional rows available: {row_count - 25}")
                    result_lines.append(f"Results saved internally to file: {filename}")
                    result = "\n".join(result_lines)

                    # Successful tool UI is intentionally not rendered by Agent in
                    # normal chat. Keep a compatibility payload only.
                    ui_component = UiComponent(
                        rich_component=SimpleTextComponent(text=""),
                        simple_component=SimpleTextComponent(text=""),
                    )

                    metadata = {
                        "row_count": row_count,
                        "columns": columns,
                        "query_type": query_type,
                        "results": results_data,
                        "output_file": filename,
                    }
            else:
                # For non-SELECT queries (INSERT, UPDATE, DELETE, etc.)
                # The SqlRunner should return a DataFrame with affected row count
                rows_affected = len(df) if not df.empty else 0
                result = (
                    f"Query executed successfully. {rows_affected} row(s) affected."
                )

                metadata = {"rows_affected": rows_affected, "query_type": query_type}
                ui_component = UiComponent(
                    rich_component=NotificationComponent(
                        type=ComponentType.NOTIFICATION, level="success", message=result
                    ),
                    simple_component=SimpleTextComponent(text=result),
                )

            return ToolResult(
                success=True,
                result_for_llm=result,
                ui_component=ui_component,
                metadata=metadata,
            )

        except Exception as e:
            error_message = f"Error executing query: {str(e)}"
            return ToolResult(
                success=False,
                result_for_llm=error_message,
                ui_component=UiComponent(
                    rich_component=NotificationComponent(
                        type=ComponentType.NOTIFICATION,
                        level="error",
                        message=error_message,
                    ),
                    simple_component=SimpleTextComponent(text=error_message),
                ),
                error=str(e),
                metadata={"error_type": "sql_error"},
            )
