"""Hozpitality AI Search V6 global retrieval engine.

Design goals:
- One fast global retrieval path for all searchable Hozpitality entities.
- PostgreSQL-first: FTS + trigram candidate retrieval, optional pgvector reranking.
- No full-table scans by the application and no LLM call for ordinary retrieval.
- Resolve master-search (content_type_id, object_id) pairs to authoritative source rows.
- Keep source metadata dynamic from PostgreSQL/Django metadata.
"""
from __future__ import annotations

import re
import time
from collections import defaultdict
from typing import Any, Optional

import psycopg2
import psycopg2.extras
from psycopg2 import sql

from .embedding_service import EmbeddingService
from .query_planner import QueryPlanner
from .search_cache import TTLSearchCache

STOPWORDS = {
    "a", "an", "and", "are", "at", "be", "by", "can", "do", "for", "from",
    "find", "get", "give", "has", "have", "how", "i", "in", "is", "it", "list",
    "me", "my", "of", "on", "or", "please", "search", "show", "some", "tell",
    "the", "to", "what", "where", "which", "who", "with", "about", "any", "all",
    "latest", "recent", "new", "news", "article", "articles", "story", "stories",
    "looking", "look", "need", "want", "would", "like", "named", "name", "called",
}

SENSITIVE_COLUMNS = {
    "password", "password_hash", "token", "access_token", "refresh_token", "secret",
    "secret_key", "api_key", "private_key", "otp", "otp_code", "session_data", "key",
}


class GlobalSearchService:
    def __init__(self, pg_config: dict[str, Any], schema_intelligence):
        self.pg_config = pg_config
        self.schema = schema_intelligence
        self.planner = QueryPlanner()
        self.embedding = EmbeddingService()
        self.cache = TTLSearchCache(
            max_items=int(__import__("os").getenv("SEARCH_CACHE_ITEMS", "512")),
            ttl_seconds=int(__import__("os").getenv("SEARCH_CACHE_TTL", "45")),
        )
        self._table_cache: Optional[dict[str, dict[str, Any]]] = None
        self._source_cache: dict[int, Optional[dict[str, Any]]] = {}
        self._search_column: Optional[str] = None
        self.last_stats: dict[str, Any] = {}

    def _connect(self):
        return psycopg2.connect(**self.pg_config)

    @staticmethod
    def _norm(value: Any) -> str:
        return re.sub(r"\s+", " ", str(value or "").strip().lower())

    @staticmethod
    def _tokens(value: str) -> list[str]:
        return [
            token for token in re.findall(r"[\w][\w'&.-]*", value.lower())
            if len(token) > 1 and token not in STOPWORDS
        ]

    def _detect_search_column(self) -> str:
        if self._search_column:
            return self._search_column
        try:
            with self._connect() as conn, conn.cursor() as cur:
                cur.execute(
                    """SELECT column_name FROM information_schema.columns
                       WHERE table_schema='public'
                         AND table_name='master_search_mastersearchindex'
                         AND column_name IN ('search_vector_v6','search_vector')
                       ORDER BY CASE column_name WHEN 'search_vector_v6' THEN 0 ELSE 1 END"""
                )
                row = cur.fetchone()
                self._search_column = row[0] if row else "search_vector"
        except Exception:
            self._search_column = "search_vector"
        return self._search_column

    def _table_metadata(self) -> dict[str, dict[str, Any]]:
        if self._table_cache is not None:
            return self._table_cache
        result: dict[str, dict[str, Any]] = {}
        query = """
            SELECT c.table_name, c.column_name, c.data_type,
                   CASE WHEN tc.constraint_type = 'PRIMARY KEY' THEN TRUE ELSE FALSE END AS is_pk
            FROM information_schema.columns c
            LEFT JOIN information_schema.key_column_usage kcu
              ON kcu.table_schema=c.table_schema AND kcu.table_name=c.table_name
             AND kcu.column_name=c.column_name
            LEFT JOIN information_schema.table_constraints tc
              ON tc.constraint_schema=kcu.constraint_schema AND tc.constraint_name=kcu.constraint_name
             AND tc.table_name=kcu.table_name
            WHERE c.table_schema='public'
            ORDER BY c.table_name, c.ordinal_position
        """
        with self._connect() as conn, conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(query)
            for row in cur.fetchall():
                entry = result.setdefault(row["table_name"], {"columns": [], "pk": []})
                entry["columns"].append(row["column_name"])
                if row.get("is_pk"):
                    entry["pk"].append(row["column_name"])
        self._table_cache = result
        return result

    @staticmethod
    def _singular(value: str) -> str:
        value = re.sub(r"[^a-z0-9]+", "", value.lower())
        if value.endswith("ies"):
            return value[:-3] + "y"
        if value.endswith("ses"):
            return value[:-2]
        if value.endswith("s") and not value.endswith("ss"):
            return value[:-1]
        return value

    def _resolve_source_table(self, content_type_id: int) -> Optional[dict[str, Any]]:
        if content_type_id in self._source_cache:
            return self._source_cache[content_type_id]

        cts = [x for x in self.schema.content_types() if int(x["id"]) == int(content_type_id)]
        if not cts:
            self._source_cache[content_type_id] = None
            return None
        ct = cts[0]
        model = self._norm(ct.get("model"))
        app_label = self._norm(ct.get("app_label"))
        compact_model = re.sub(r"[^a-z0-9]+", "", model)
        singular_model = self._singular(compact_model)
        tables = self._table_metadata()

        candidates: list[tuple[int, str]] = []
        for table, meta in tables.items():
            if not meta.get("pk"):
                continue
            compact_table = re.sub(r"[^a-z0-9]+", "", table.lower())
            score = 0
            if compact_table == singular_model:
                score += 160
            if self._singular(compact_table) == singular_model:
                score += 130
            if compact_table.endswith(singular_model):
                score += 90
            if model and model in table.lower():
                score += 25
            if app_label and app_label in table.lower():
                score += 15
            # Prefer real entity tables over many-to-many/through tables.
            if "through" in compact_table or compact_table.endswith("_through"):
                score -= 60
            if score:
                candidates.append((score, table))

        if not candidates:
            self._source_cache[content_type_id] = None
            return None
        candidates.sort(key=lambda item: (-item[0], len(item[1])))
        table = candidates[0][1]
        resolved = {
            "table": table,
            "pk": tables[table]["pk"][0],
            "columns": tables[table]["columns"],
            "content_type": ct,
        }
        self._source_cache[content_type_id] = resolved
        return resolved

    @staticmethod
    def _safe_columns(columns: list[str], max_columns: int = 28) -> list[str]:
        safe = []
        for col in columns:
            low = col.lower()
            if low in SENSITIVE_COLUMNS or any(x in low for x in ("password", "secret", "token", "session")):
                continue
            if low in {"embedding", "search_vector", "search_vector_v6"}:
                continue
            safe.append(col)
        return safe[:max_columns]

    def _fetch_source_rows(self, hits: list[dict[str, Any]]) -> dict[tuple[int, int], dict[str, Any]]:
        grouped: dict[tuple[str, str, int], list[int]] = defaultdict(list)
        for hit in hits:
            source = self._resolve_source_table(int(hit["content_type_id"]))
            if source:
                grouped[(source["table"], source["pk"], int(hit["content_type_id"]))].append(int(hit["object_id"]))

        if not grouped:
            return {}
        output: dict[tuple[int, int], dict[str, Any]] = {}
        with self._connect() as conn, conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            for (table, pk, ct_id), object_ids in grouped.items():
                source = self._resolve_source_table(ct_id)
                if not source:
                    continue
                cols = self._safe_columns(source["columns"])
                if not cols:
                    continue
                query = sql.SQL(
                    "SELECT {cols} FROM public.{table} WHERE {pk} = ANY(%s) LIMIT 50"
                ).format(
                    cols=sql.SQL(", ").join(sql.Identifier(c) for c in cols),
                    table=sql.Identifier(table),
                    pk=sql.Identifier(pk),
                )
                try:
                    cur.execute(query, [list(dict.fromkeys(object_ids))])
                    for row in cur.fetchall():
                        value = row.get(pk)
                        if value is not None:
                            output[(ct_id, int(value))] = dict(row)
                except Exception as exc:
                    conn.rollback()
                    print(f"V6 source enrichment skipped {table}: {exc}")
        return output

    @staticmethod
    def _display(value: Any, limit: int = 260) -> str:
        if value is None:
            return ""
        text = re.sub(r"\s+", " ", str(value)).strip()
        return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"

    def _format_hit(self, hit: dict[str, Any], source: Optional[dict[str, Any]]) -> list[str]:
        title = hit.get("title") or "Untitled result"
        model = hit.get("model") or "record"
        lines = [f"**{title}**", f"- Type: {model}"]
        if hit.get("location_text"):
            lines.append(f"- Location: {self._display(hit['location_text'], 160)}")
        if hit.get("category_text"):
            lines.append(f"- Category: {self._display(hit['category_text'], 160)}")
        if hit.get("user_name"):
            lines.append(f"- Person/Owner: {self._display(hit['user_name'], 160)}")
        if hit.get("slug"):
            lines.append(f"- Slug: `{self._display(hit['slug'], 180)}`")
        if source:
            useful = [
                "name", "first_name", "last_name", "job_title", "job_city", "job_status",
                "job_role", "resume_title", "current_company_text", "company_name", "website_link",
                "website", "status", "event_type", "price", "condition", "description", "question",
                "answer", "details", "created_at", "start_datetime", "end_datetime", "title", "sub_title",
                "content", "location", "city", "country", "award_title", "award_description",
            ]
            used = set()
            for name in useful:
                for col in source["columns"]:
                    if col.lower() == name.lower() and col not in used:
                        value = source.get(col)
                        if value not in (None, ""):
                            lines.append(f"- {col.replace('_', ' ').title()}: {self._display(value)}")
                            used.add(col)
                        break
        return lines

    def _live_clause(self, historical: bool) -> str:
        if historical:
            return "TRUE"
        return "(si.is_live = TRUE OR si.is_live IS NULL) AND (si.expires_at IS NULL OR si.expires_at >= CURRENT_TIMESTAMP)"

    def _build_candidate_query(self, tokens: list[str], phrase: str, content_type_id: Optional[int], historical: bool, limit: int, query_vector: Optional[list[float]]):
        search_col = self._detect_search_column()
        tsquery = " & ".join(tokens[:12]) if tokens else phrase
        like_phrase = f"%{phrase}%"
        trigram_threshold = 0.18 if len(phrase) >= 5 else 0.25
        ctes = [
            f"fts AS (SELECT si.id, ts_rank_cd(si.{search_col}, plainto_tsquery('simple', unaccent(%s))) AS fts_score FROM master_search_mastersearchindex si WHERE {self._live_clause(historical)} AND si.{search_col} @@ plainto_tsquery('simple', unaccent(%s)) ORDER BY fts_score DESC LIMIT 100)",
            "trgm AS (SELECT si.id, GREATEST(similarity(si.title, %s), similarity(si.user_name, %s), similarity(si.category_text, %s), similarity(si.location_text, %s), similarity(si.slug, %s)) AS trgm_score FROM master_search_mastersearchindex si WHERE "
            + self._live_clause(historical)
            + " AND (si.title % %s OR si.user_name % %s OR si.category_text % %s OR si.location_text % %s OR si.slug % %s) LIMIT 100)",
        ]
        params: list[Any] = [tsquery, tsquery, phrase, phrase, phrase, phrase, phrase, phrase, phrase, phrase, phrase, phrase]
        if query_vector:
            ctes.append(
                "vec AS (SELECT si.id, 1 - (si.embedding <=> %s::vector) AS vec_score FROM master_search_mastersearchindex si WHERE "
                + self._live_clause(historical)
                + " AND si.embedding IS NOT NULL ORDER BY si.embedding <=> %s::vector LIMIT 100)"
            )
            vec = "[" + ",".join(f"{float(x):.8f}" for x in query_vector) + "]"
            params.extend([vec, vec])

        ctes.append("candidates AS (SELECT id FROM fts UNION SELECT id FROM trgm" + (" UNION SELECT id FROM vec" if query_vector else "") + ")")
        type_filter = ""
        extra_params: list[Any] = []
        if content_type_id is not None:
            type_filter = " AND si.content_type_id = %s"
            extra_params.append(content_type_id)

        sql_text = f"""
            WITH {', '.join(ctes)}
            SELECT
                si.id, si.object_id, si.title, si.location_text, si.category_text,
                si.ai_keywords, si.is_live, si.created_at, si.content_type_id,
                si.expires_at, si.user_name, si.content, si.slug,
                ct.app_label, ct.model,
                COALESCE(fts.fts_score, 0) AS fts_score,
                COALESCE(trgm.trgm_score, 0) AS trgm_score,
                {"COALESCE(vec.vec_score, 0)" if query_vector else "0"} AS vec_score,
                (
                    CASE WHEN lower(COALESCE(si.title,'')) = lower(%s) THEN 200 ELSE 0 END +
                    CASE WHEN position(lower(%s) in lower(COALESCE(si.title,''))) > 0 THEN 100 ELSE 0 END +
                    CASE WHEN position(lower(%s) in lower(COALESCE(si.user_name,''))) > 0 THEN 85 ELSE 0 END +
                    CASE WHEN position(lower(%s) in lower(COALESCE(si.category_text,''))) > 0 THEN 70 ELSE 0 END +
                    COALESCE(fts.fts_score,0) * 35 +
                    COALESCE(trgm.trgm_score,0) * 35 +
                    {"COALESCE(vec.vec_score,0) * 30" if query_vector else "0"}
                ) AS relevance
            FROM candidates c
            JOIN master_search_mastersearchindex si ON si.id = c.id
            JOIN django_content_type ct ON ct.id = si.content_type_id
            LEFT JOIN fts ON fts.id = si.id
            LEFT JOIN trgm ON trgm.id = si.id
            {"LEFT JOIN vec ON vec.id = si.id" if query_vector else ""}
            WHERE {self._live_clause(historical)} {type_filter}
            ORDER BY relevance DESC, si.created_at DESC NULLS LAST, si.id DESC
            LIMIT %s
        """
        params.extend([phrase, phrase, phrase, phrase])
        params.extend(extra_params)
        params.append(limit)
        return sql_text, params

    def search_hits(self, message: str, limit: int = 10) -> list[dict[str, Any]]:
        plan = self.planner.plan(message, limit)
        if plan.analytics:
            return []
        tokens = self._tokens(plan.normalized)
        phrase = " ".join(tokens[:12]).strip() or plan.normalized
        if len(phrase) < 2:
            return []

        ct = self.schema.resolve_content_type(plan.normalized)
        content_type_id = ct.id if ct else None
        query_vector = self.embedding.encode(phrase)
        cache_key = self.cache.key(plan.normalized, str(content_type_id or "all"), str(plan.historical), str(limit), "v6")
        cached = self.cache.get(cache_key)
        if cached is not None:
            self.last_stats = {"cache_hit": True, "strategy": "GLOBAL_SEARCH", "count": len(cached)}
            return cached

        started = time.perf_counter()
        query, params = self._build_candidate_query(
            tokens, phrase, content_type_id, plan.historical, max(1, min(limit, 20)), query_vector
        )
        try:
            with self._connect() as conn, conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(query, params)
                hits = [dict(row) for row in cur.fetchall()]
        except Exception as exc:
            # If vector search is unavailable/malformed, retry lexical-only.
            if query_vector:
                query, params = self._build_candidate_query(
                    tokens, phrase, content_type_id, plan.historical, max(1, min(limit, 20)), None
                )
                with self._connect() as conn, conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    cur.execute(query, params)
                    hits = [dict(row) for row in cur.fetchall()]
            else:
                print(f"V6 global search failed: {exc}")
                return []

        self.cache.set(cache_key, hits)
        self.last_stats = {
            "cache_hit": False,
            "strategy": plan.strategy,
            "content_type": plan.entity_type,
            "count": len(hits),
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            "vector": bool(query_vector),
        }
        return hits

    def search(self, message: str, limit: int = 10) -> Optional[str]:
        hits = self.search_hits(message, limit)
        if not hits:
            return None
        source_rows = self._fetch_source_rows(hits)
        lines = [f"I found {len(hits)} matching Hozpitality records.", ""]
        for index, hit in enumerate(hits, 1):
            lines.append(f"### {index}")
            lines.extend(self._format_hit(hit, source_rows.get((int(hit["content_type_id"]), int(hit["object_id"])))) )
            lines.append("")
        return "\n".join(lines).strip()
