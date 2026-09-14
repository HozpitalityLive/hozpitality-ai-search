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
        return (
            "(si.is_live = TRUE OR si.is_live IS NULL) "
            "AND (si.expires_at IS NULL OR si.expires_at >= CURRENT_TIMESTAMP)"
        )


    def _classify_location_tokens(self, tokens: list[str]) -> list[str]:
        """Classify geography tokens from real word matches in location_text."""
        if not tokens:
            return []

        unique = list(dict.fromkeys(t for t in tokens if len(t) >= 3))[:12]
        if not unique:
            return []

        branches = []
        params: list[Any] = []
        for token in unique:
            # Normalize punctuation in location_text to spaces, then search
            # for a whole word. This avoids pg_trgm false positives such as
            # "chef" being treated as a location because of bad data.
            branches.append(
                f"""
                SELECT %s AS token,
                       COUNT(*) AS location_hits
                FROM master_search_mastersearchindex si
                WHERE {self._live_clause(False)}
                  AND position(
                      ' ' || lower(%s) || ' '
                      IN ' ' || regexp_replace(
                          lower(COALESCE(si.location_text, '')),
                          '[^a-z0-9]+', ' ', 'g'
                      ) || ' '
                  ) > 0
                """
            )
            params.extend([token, token])

        query = " UNION ALL ".join(branches)
        try:
            with self._connect() as conn, conn.cursor(
                cursor_factory=psycopg2.extras.RealDictCursor
            ) as cur:
                cur.execute(query, params)
                rows = cur.fetchall()
        except Exception as exc:
            print(f"V6 location classification skipped: {exc}")
            return []

        valid = {
            str(row["token"]).lower()
            for row in rows
            if int(row["location_hits"] or 0) >= 3
        }
        return [token for token in unique if token.lower() in valid]

    def _build_candidate_query(
        self,
        keyword_tokens: list[str],
        keyword_phrase: str,
        location_tokens: list[str],
        location_phrase: str,
        content_type_id: Optional[int],
        historical: bool,
        limit: int,
        query_vector: Optional[list[float]],
    ):
        """Build a deterministic, bounded hybrid query.

        Explicit locations are HARD constraints, not merely ranking signals.
        This is essential for queries such as "chef jobs Dubai": the candidate
        pool must contain records matching both the semantic keyword and Dubai.
        """
        search_col = self._detect_search_column()
        keyword_phrase = (keyword_phrase or "").strip()
        location_phrase = (location_phrase or "").strip()
        live = self._live_clause(historical)
        lim = max(1, min(int(limit), 20))
        candidate_limit = max(100, min(lim * 30, 500))

        params: list[Any] = []
        ctes: list[str] = []

        type_clause = ""
        if content_type_id is not None:
            type_clause = " AND si.content_type_id = %s"

        def location_condition(alias="si"):
            parts = []
            for token in location_tokens[:6]:
                pattern = f"%{token}%"
                parts.append(
                    f"(lower(COALESCE({alias}.location_text,'')) LIKE lower(%s)"
                    f" OR lower(COALESCE({alias}.title,'')) LIKE lower(%s)"
                    f" OR lower(COALESCE({alias}.content,'')) LIKE lower(%s)"
                    f" OR lower(COALESCE({alias}.ai_keywords,'')) LIKE lower(%s)"
                    f" OR lower(COALESCE({alias}.slug,'')) LIKE lower(%s))"
                )
            return " OR ".join(parts) if parts else "TRUE"

        # FTS candidate pool. When a location is explicit, enforce it INSIDE
        # this CTE so the top-N keyword candidates cannot crowd Dubai out.
        if keyword_phrase:
            fts_where = f"""{live}{type_clause}
                      AND si.{search_col} @@ websearch_to_tsquery(
                          'simple', unaccent(%s)
                      )"""
            params_fts: list[Any] = []
            if content_type_id is not None:
                params_fts.append(content_type_id)
            params_fts.append(keyword_phrase)
            if location_tokens:
                fts_where += f" AND ({location_condition()})"
                for token in location_tokens[:6]:
                    params_fts.extend([f"%{token}%"] * 5)
            ctes.append(f"""
                fts AS (
                    SELECT si.id,
                           ts_rank_cd(
                               si.{search_col},
                               websearch_to_tsquery('simple', unaccent(%s))
                           ) AS fts_score
                    FROM master_search_mastersearchindex si
                    WHERE {live}
                      {type_clause}
                      AND si.{search_col} @@ websearch_to_tsquery(
                          'simple', unaccent(%s)
                      )
                      {f"AND ({location_condition()})" if location_tokens else ""}
                    ORDER BY fts_score DESC, si.id DESC
                    LIMIT {candidate_limit}
                )
            """)
            # SELECT rank expression params, then WHERE params, in exact order.
            params.extend([keyword_phrase])
            if content_type_id is not None:
                params.extend([content_type_id])
            params.extend([keyword_phrase])
            if location_tokens:
                for token in location_tokens[:6]:
                    params.extend([f"%{token}%"] * 5)
        else:
            ctes.append("fts AS (SELECT NULL::bigint AS id, 0::float AS fts_score WHERE FALSE)")

        # Trigram candidate pool. Search each semantic token independently so
        # misspellings such as "restarant" can still retrieve "restaurant".
        trigram_terms = [t for t in keyword_tokens if len(t) >= 2][:8]
        if not trigram_terms and keyword_phrase:
            trigram_terms = [keyword_phrase[:80]]

        if trigram_terms:
            conds = []
            score_parts = []
            score_params: list[Any] = []
            where_params: list[Any] = []
            for token in trigram_terms:
                conds.append(
                    "(si.title %% %s OR si.user_name %% %s OR "
                    "si.category_text %% %s OR si.slug %% %s)"
                )
                where_params.extend([token] * 4)
                score_parts.append(
                    "GREATEST(similarity(COALESCE(si.title,''), %s),"
                    "similarity(COALESCE(si.user_name,''), %s),"
                    "similarity(COALESCE(si.category_text,''), %s),"
                    "similarity(COALESCE(si.slug,''), %s))"
                )
                score_params.extend([token] * 4)

            loc_sql = f" AND ({location_condition()})" if location_tokens else ""
            ctes.append(f"""
                trgm AS (
                    SELECT si.id,
                           GREATEST({",".join(score_parts)}) AS trgm_score
                    FROM master_search_mastersearchindex si
                    WHERE {live}
                      {type_clause}
                      AND ({" OR ".join(conds)})
                      {loc_sql}
                    ORDER BY trgm_score DESC, si.id DESC
                    LIMIT {candidate_limit}
                )
            """)
            # score expression appears before WHERE expressions.
            params.extend(score_params)
            if content_type_id is not None:
                params.extend([content_type_id])
            params.extend(where_params)
            if location_tokens:
                for token in location_tokens[:6]:
                    params.extend([f"%{token}%"] * 5)
        else:
            ctes.append("trgm AS (SELECT NULL::bigint AS id, 0::float AS trgm_score WHERE FALSE)")

        # Optional vector candidate pool. Keep it bounded and only include it
        # when explicitly enabled.
        if query_vector:
            vec = "[" + ",".join(f"{float(x):.8f}" for x in query_vector) + "]"
            ctes.append(f"""
                vec AS (
                    SELECT si.id,
                           1 - (si.embedding <=> %s::vector) AS vec_score
                    FROM master_search_mastersearchindex si
                    WHERE {live}
                      {type_clause}
                      AND si.embedding IS NOT NULL
                      {f"AND ({location_condition()})" if location_tokens else ""}
                    ORDER BY si.embedding <=> %s::vector
                    LIMIT {candidate_limit}
                )
            """)
            params.append(vec)
            if content_type_id is not None:
                params.append(content_type_id)
            if location_tokens:
                for token in location_tokens[:6]:
                    params.extend([f"%{token}%"] * 5)
            params.append(vec)
        else:
            ctes.append("vec AS (SELECT NULL::bigint AS id, 0::float AS vec_score WHERE FALSE)")

        ctes.append("""
            candidates AS (
                SELECT id FROM fts
                UNION
                SELECT id FROM trgm
                UNION
                SELECT id FROM vec
            )
        """)

        # Final SELECT parameters are kept at the end so parameter ordering is
        # straightforward and independent of the CTE construction.
        title_exact = keyword_phrase or ""
        title_like = f"%{keyword_phrase}%" if keyword_phrase else ""
        loc_like = f"%{location_phrase}%" if location_phrase else ""

        select_sql = f"""
            WITH {", ".join(ctes)}
            SELECT
                si.id,
                si.object_id,
                si.title,
                si.location_text,
                si.category_text,
                si.ai_keywords,
                si.is_live,
                si.created_at,
                si.content_type_id,
                si.expires_at,
                si.user_name,
                si.content,
                si.slug,
                ct.app_label,
                ct.model,
                COALESCE(fts.fts_score, 0) AS fts_score,
                COALESCE(trgm.trgm_score, 0) AS trgm_score,
                0::float AS loc_score,
                COALESCE(vec.vec_score, 0) AS vec_score,
                (
                    CASE WHEN %s <> '' AND lower(COALESCE(si.title,'')) = lower(%s)
                         THEN 220 ELSE 0 END
                    + CASE WHEN %s <> '' AND lower(COALESCE(si.title,'')) LIKE lower(%s)
                           THEN 100 ELSE 0 END
                    + CASE WHEN %s <> '' AND lower(COALESCE(si.location_text,'')) LIKE lower(%s)
                           THEN 120 ELSE 0 END
                    + COALESCE(fts.fts_score, 0) * 70
                    + COALESCE(trgm.trgm_score, 0) * 50
                    + CASE WHEN %s <> '' AND lower(COALESCE(si.location_text,'')) LIKE lower(%s)
                           THEN 110 ELSE 0 END
                    + COALESCE(vec.vec_score, 0) * 30
                    + CASE WHEN %s = TRUE AND si.content_type_id = %s THEN 35 ELSE 0 END
                ) AS relevance
            FROM candidates c
            JOIN master_search_mastersearchindex si ON si.id = c.id
            JOIN django_content_type ct ON ct.id = si.content_type_id
            LEFT JOIN fts ON fts.id = si.id
            LEFT JOIN trgm ON trgm.id = si.id
            LEFT JOIN vec ON vec.id = si.id
            WHERE {live}
              {type_clause}
              {f"AND ({location_condition()})" if location_tokens else ""}
            ORDER BY relevance DESC, si.created_at DESC NULLS LAST, si.id DESC
            LIMIT %s
        """

        # Final SELECT expression parameters.
        params.extend([
            keyword_phrase, title_exact,
            keyword_phrase, title_like,
            location_phrase, loc_like,
            location_phrase, loc_like,
            content_type_id is not None, content_type_id or 0,
        ])
        # Final WHERE type/location parameters.
        if content_type_id is not None:
            params.append(content_type_id)
        if location_tokens:
            for token in location_tokens[:6]:
                params.extend([f"%{token}%"] * 5)
        params.append(lim)

        return select_sql, params

    def _vector_enabled(self) -> bool:
        import os
        return os.getenv("SEARCH_VECTOR_ENABLED", "false").strip().lower() in {
            "1", "true", "yes", "on"
        }

    def _entity_stopwords(self, entity_type: str) -> set[str]:
        aliases = {
            "job": {"job", "jobs", "vacancy", "vacancies", "career", "careers",
                    "employment", "opening", "openings", "position", "positions"},
            "professional": {"professional", "professionals", "candidate", "candidates",
                              "profile", "profiles", "resume", "cv"},
            "company": {"company", "companies", "employer", "employers",
                        "organization", "organisation", "organisations"},
            "article": {"article", "articles", "news", "blog", "blogs", "story", "stories"},
            "event": {"event", "events", "conference", "conferences", "expo", "exhibition",
                      "exhibitions"},
            "product": {"product", "products", "marketplace", "supplier", "suppliers"},
            "faq": {"faq", "faqs", "question", "questions", "help"},
            "award": {"award", "awards", "recognition", "winner", "winners",
                      "nomination", "nominations"},
        }
        return aliases.get(entity_type, set())

    def search_hits(self, message: str, limit: int = 10) -> list[dict[str, Any]]:
        plan = self.planner.plan(message, limit)
        if plan.analytics:
            self.last_stats = {"strategy": "ANALYTICS", "count": 0}
            return []

        entity_type = self._norm(getattr(plan, "entity_type", "") or "")
        raw_tokens = self._tokens(plan.normalized)
        entity_words = self._entity_stopwords(entity_type)
        semantic_tokens = [t for t in raw_tokens if t not in entity_words]

        # Dynamically identify locations from the master index. The remaining
        # tokens are the semantic search terms. This fixes "chef jobs Dubai":
        # entity=job, keywords=chef, location=Dubai.
        location_tokens = self._classify_location_tokens(semantic_tokens)
        keyword_tokens = [t for t in semantic_tokens if t not in set(location_tokens)]

        # Keep at least one search term for pure typo/global queries.
        if not keyword_tokens and not location_tokens:
            keyword_tokens = semantic_tokens[:12]

        keyword_phrase = " ".join(keyword_tokens[:12]).strip()
        location_phrase = " ".join(location_tokens[:6]).strip()
        if not keyword_phrase and not location_phrase:
            self.last_stats = {
                "strategy": "GLOBAL_SEARCH",
                "count": 0,
                "reason": "query_too_short",
            }
            return []

        content_type_id: Optional[int] = None
        if entity_type:
            ct = self.schema.resolve_content_type(entity_type)
            if ct:
                content_type_id = int(ct.id)

        query_vector: Optional[list[float]] = None
        if self._vector_enabled():
            try:
                vector_text = " ".join(x for x in (keyword_tokens + location_tokens) if x)
                query_vector = self.embedding.encode(vector_text)
            except Exception as exc:
                print(f"V6 vector retrieval disabled for this request: {exc}")

        cache_key = self.cache.key(
            plan.normalized,
            str(content_type_id or "all"),
            str(plan.historical),
            str(limit),
            "v6.3",
        )
        cached = self.cache.get(cache_key)
        if cached is not None:
            self.last_stats = {
                "cache_hit": True,
                "strategy": plan.strategy,
                "content_type": entity_type or None,
                "keywords": keyword_tokens,
                "location": location_tokens,
                "count": len(cached),
                "vector": bool(query_vector),
            }
            return cached

        started = time.perf_counter()

        def execute(candidate_vector):
            query, params = self._build_candidate_query(
                keyword_tokens=keyword_tokens,
                keyword_phrase=keyword_phrase,
                location_tokens=location_tokens,
                location_phrase=location_phrase,
                content_type_id=content_type_id,
                historical=plan.historical,
                limit=max(1, min(int(limit), 20)),
                query_vector=candidate_vector,
            )
            with self._connect() as conn, conn.cursor(
                cursor_factory=psycopg2.extras.RealDictCursor
            ) as cur:
                cur.execute(query, params)
                return [dict(row) for row in cur.fetchall()]

        try:
            hits = execute(query_vector)
        except Exception as exc:
            if query_vector is not None:
                print(f"V6 vector search failed; retrying lexical search: {exc}")
                try:
                    hits = execute(None)
                    query_vector = None
                except Exception as lexical_exc:
                    print(f"V6 lexical search failed: {lexical_exc}")
                    self.last_stats = {
                        "cache_hit": False,
                        "strategy": plan.strategy,
                        "count": 0,
                        "error": str(lexical_exc),
                    }
                    return []
            else:
                print(f"V6 global search failed: {exc}")
                self.last_stats = {
                    "cache_hit": False,
                    "strategy": plan.strategy,
                    "count": 0,
                    "error": str(exc),
                }
                return []

        self.cache.set(cache_key, hits)
        self.last_stats = {
            "cache_hit": False,
            "strategy": plan.strategy,
            "content_type": entity_type or None,
            "keywords": keyword_tokens,
            "location": location_tokens,
            "count": len(hits),
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            "vector": bool(query_vector),
            "fts_trigram": True,
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
            lines.extend(
                self._format_hit(
                    hit,
                    source_rows.get(
                        (int(hit["content_type_id"]), int(hit["object_id"]))
                    ),
                )
            )
            lines.append("")
        return "\n".join(lines).strip()
