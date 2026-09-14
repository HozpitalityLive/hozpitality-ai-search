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
        """Identify likely geography tokens from indexed location data.

        This is data-driven rather than a hard-coded city list.  A token is
        treated as a location only when it has meaningful coverage in
        ``location_text``.  The coverage test is deliberately conservative so
        words such as ``chef`` cannot become a location merely because a few
        malformed records contain them in their location field.
        """
        if not tokens:
            return []

        unique = list(dict.fromkeys(t for t in tokens if len(t) >= 3))[:12]
        if not unique:
            return []

        branches = []
        params: list[Any] = []
        for token in unique:
            branches.append(
                f"""
                SELECT %s AS token,
                       COUNT(*) FILTER (
                           WHERE si.location_text %% %s
                       ) AS location_hits,
                       COUNT(*) FILTER (
                           WHERE si.title %% %s
                              OR si.ai_keywords %% %s
                              OR si.content %% %s
                       ) AS semantic_hits
                FROM master_search_mastersearchindex si
                WHERE {self._live_clause(False)}
                  AND (
                      si.location_text %% %s
                      OR si.title %% %s
                      OR si.ai_keywords %% %s
                  )
                """
            )
            # token label + location filter + 3 semantic comparisons + WHERE
            params.extend([token, token, token, token, token, token, token, token])

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

        result: list[str] = []
        for row in rows:
            token = str(row["token"]).lower()
            location_hits = int(row["location_hits"] or 0)
            semantic_hits = int(row["semantic_hits"] or 0)

            # Require real location coverage.  The relative test handles both
            # large cities (Dubai) and smaller geographies while the absolute
            # floor prevents isolated/malformed location values from winning.
            if location_hits >= 10 and (
                location_hits >= semantic_hits * 0.02
                or location_hits >= 50
            ):
                result.append(token)

        return [token for token in unique if token.lower() in set(result)]

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
        """Build the bounded V6 hybrid candidate/reranking query.

        Query understanding is deliberately separated from retrieval:
        * entity words (e.g. ``jobs``) are not searched as keywords once the
          entity is resolved;
        * location tokens (e.g. ``Dubai``) are retrieved/scored separately;
        * FTS handles normal terms;
        * trigram handles typos such as ``restarant``;
        * pgvector is optional;
        * expensive scoring happens only on bounded candidates.
        """
        search_col = self._detect_search_column()
        keyword_phrase = keyword_phrase.strip() or " ".join(keyword_tokens).strip()
        location_phrase = location_phrase.strip()

        fts_limit = max(100, min(int(limit) * 20, 400))
        trgm_limit = max(100, min(int(limit) * 20, 400))
        loc_limit = max(100, min(int(limit) * 20, 300))
        vector_limit = max(100, min(int(limit) * 20, 300))

        ctes: list[str] = []
        params: list[Any] = []

        # FTS is only applied to semantic keywords, not entity/location words.
        if keyword_phrase:
            ctes.append(
                f"""
                fts AS (
                    SELECT si.id,
                           ts_rank_cd(
                               si.{search_col},
                               websearch_to_tsquery('simple', unaccent(%s))
                           ) AS fts_score
                    FROM master_search_mastersearchindex si
                    WHERE {self._live_clause(historical)}
                      AND si.{search_col} @@ websearch_to_tsquery(
                          'simple', unaccent(%s)
                      )
                    ORDER BY fts_score DESC, si.id DESC
                    LIMIT {fts_limit}
                )
                """
            )
            params.extend([keyword_phrase, keyword_phrase])
        else:
            ctes.append("fts AS (SELECT NULL::bigint AS id, 0::float AS fts_score WHERE FALSE)")

        # Trigram retrieval is token-aware. This is important for typo queries
        # and for multiword phrases where whole-phrase similarity is weak.
        trigram_terms = [t for t in keyword_tokens if len(t) >= 2]
        if not trigram_terms and keyword_phrase:
            trigram_terms = [keyword_phrase]

        trgm_conditions = []
        trgm_score_args = []
        for token in trigram_terms[:8]:
            trgm_conditions.append(
                "(si.title %% %s OR si.user_name %% %s OR si.category_text %% %s "
                "OR si.slug %% %s)"
            )
            trgm_score_args.extend([token, token, token, token])

        # GREATEST of token/field similarities; parameters are repeated because
        # psycopg2 does not support named parameters in this positional query.
        score_terms = []
        score_params: list[Any] = []
        for token in trigram_terms[:8]:
            score_terms.append(
                "GREATEST("
                "similarity(COALESCE(si.title,''), %s),"
                "similarity(COALESCE(si.user_name,''), %s),"
                "similarity(COALESCE(si.category_text,''), %s),"
                "similarity(COALESCE(si.slug,''), %s)"
                ")"
            )
            score_params.extend([token, token, token, token])

        trgm_where = " OR ".join(trgm_conditions) if trgm_conditions else "FALSE"
        trgm_score = "GREATEST(" + ",".join(score_terms) + ")" if score_terms else "0"

        ctes.append(
            f"""
            trgm AS (
                SELECT si.id,
                       {trgm_score} AS trgm_score
                FROM master_search_mastersearchindex si
                WHERE {self._live_clause(historical)}
                  AND ({trgm_where})
                ORDER BY trgm_score DESC, si.id DESC
                LIMIT {trgm_limit}
            )
            """
        )
        # score parameters appear before WHERE/operator parameters.
        params.extend(score_params)
        for token in trigram_terms[:8]:
            params.extend([token, token, token, token])

        # Separate location candidate pool. It prevents a common word such as
        # "chef" from crowding Dubai jobs out of the bounded lexical pool.
        loc_where_parts = []
        loc_params: list[Any] = []
        for token in location_tokens[:6]:
            loc_where_parts.append("si.location_text %% %s")
            loc_params.append(token)

        if loc_where_parts:
            loc_score = "GREATEST(" + ",".join(
                ["similarity(COALESCE(si.location_text,''), %s)"] * len(location_tokens[:6])
            ) + ")"
            ctes.append(
                f"""
                loc AS (
                    SELECT si.id,
                           {loc_score} AS loc_score
                    FROM master_search_mastersearchindex si
                    WHERE {self._live_clause(historical)}
                      AND ({' OR '.join(loc_where_parts)})
                    ORDER BY loc_score DESC, si.id DESC
                    LIMIT {loc_limit}
                )
                """
            )
            for token in location_tokens[:6]:
                params.append(token)
            params.extend(loc_params)
        else:
            ctes.append("loc AS (SELECT NULL::bigint AS id, 0::float AS loc_score WHERE FALSE)")

        if query_vector:
            vec = "[" + ",".join(f"{float(x):.8f}" for x in query_vector) + "]"
            ctes.append(
                f"""
                vec AS (
                    SELECT si.id,
                           1 - (si.embedding <=> %s::vector) AS vec_score
                    FROM master_search_mastersearchindex si
                    WHERE {self._live_clause(historical)}
                      AND si.embedding IS NOT NULL
                    ORDER BY si.embedding <=> %s::vector
                    LIMIT {vector_limit}
                )
                """
            )
            params.extend([vec, vec])

        union_parts = ["SELECT id FROM fts", "SELECT id FROM trgm", "SELECT id FROM loc"]
        if query_vector:
            union_parts.append("SELECT id FROM vec")
        ctes.append("candidates AS (" + " UNION ".join(union_parts) + ")")

        type_filter = ""
        extra_params: list[Any] = []
        if content_type_id is not None:
            type_filter = " AND si.content_type_id = %s"
            extra_params.append(content_type_id)

        # Per-token lexical and location boosts. These are calculated only on
        # the bounded UNION candidate set.
        keyword_boost_parts = []
        keyword_boost_params: list[Any] = []
        for token in keyword_tokens[:8]:
            keyword_boost_parts.append(
                "CASE WHEN lower(COALESCE(si.title,'')) LIKE lower(%s) THEN 55 ELSE 0 END"
            )
            keyword_boost_params.append(f"%{token}%")

        location_boost_parts = []
        location_boost_params: list[Any] = []
        for token in location_tokens[:6]:
            location_boost_parts.append(
                "CASE WHEN lower(COALESCE(si.location_text,'')) LIKE lower(%s) THEN 95 ELSE 0 END"
            )
            location_boost_params.append(f"%{token}%")

        exact_title_params = [keyword_phrase, keyword_phrase] if keyword_phrase else ["", ""]
        location_phrase_param = [location_phrase] if location_phrase else [""]

        sql_text = f"""
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
                COALESCE(loc.loc_score, 0) AS loc_score,
                {"COALESCE(vec.vec_score, 0)" if query_vector else "0"} AS vec_score,
                (
                    CASE WHEN lower(COALESCE(si.title,'')) = lower(%s)
                         THEN 220 ELSE 0 END
                    + CASE WHEN lower(COALESCE(si.title,'')) LIKE lower(%s)
                           THEN 100 ELSE 0 END
                    + CASE WHEN %s <> ''
                              AND lower(COALESCE(si.location_text,'')) LIKE lower(%s)
                           THEN 120 ELSE 0 END
                    + {" + ".join(keyword_boost_parts) if keyword_boost_parts else "0"}
                    + {" + ".join(location_boost_parts) if location_boost_parts else "0"}
                    + COALESCE(fts.fts_score, 0) * 55
                    + COALESCE(trgm.trgm_score, 0) * 42
                    + COALESCE(loc.loc_score, 0) * 65
                    {" + COALESCE(vec.vec_score, 0) * 30" if query_vector else ""}
                    + CASE WHEN %s = TRUE AND si.content_type_id = %s THEN 35 ELSE 0 END
                ) AS relevance
            FROM candidates c
            JOIN master_search_mastersearchindex si ON si.id = c.id
            JOIN django_content_type ct ON ct.id = si.content_type_id
            LEFT JOIN fts ON fts.id = si.id
            LEFT JOIN trgm ON trgm.id = si.id
            LEFT JOIN loc ON loc.id = si.id
            {"LEFT JOIN vec ON vec.id = si.id" if query_vector else ""}
            WHERE {self._live_clause(historical)}
              {type_filter}
            ORDER BY relevance DESC, si.created_at DESC NULLS LAST, si.id DESC
            LIMIT %s
        """

        params.extend(exact_title_params)
        if location_phrase:
            params.extend([location_phrase, f"%{location_phrase}%"])
        else:
            params.extend(["", ""])
        params.extend(keyword_boost_params)
        params.extend(location_boost_params)
        params.extend([content_type_id is not None, content_type_id or 0])
        params.extend(extra_params)
        params.append(limit)
        return sql_text, params

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
            "v6.1",
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
