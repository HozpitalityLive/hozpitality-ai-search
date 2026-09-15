#!/usr/bin/env bash
set -euo pipefail

: "${POSTGRES_HOST:?POSTGRES_HOST is required}"
: "${POSTGRES_USER:?POSTGRES_USER is required}"
: "${POSTGRES_DATABASE:?POSTGRES_DATABASE is required}"

PSQL=(psql -h "$POSTGRES_HOST" -U "$POSTGRES_USER" -d "$POSTGRES_DATABASE")

echo '== 1/4 Foundation =='
"${PSQL[@]}" -f sql/010_v6_search_foundation.sql

echo '== 2/5 Canonical AI search document =='
echo 'Run the following separately after checking production load:'
echo '  psql -h "$POSTGRES_HOST" -U "$POSTGRES_USER" -d "$POSTGRES_DATABASE" -f sql/013_v6_ai_search_document.sql'
echo '  python scripts/backfill_ai_search_text.py --limit 1000 --batch-size 500 --sleep 0.10 --confirm'
echo '  python scripts/backfill_ai_search_text.py --batch-size 500 --sleep 0.05 --confirm'
echo '  python scripts/backfill_search_vector_v6.py --limit 1000 --batch-size 500 --sleep 0.20 --confirm'
echo '  python scripts/backfill_search_vector_v6.py --batch-size 1000 --sleep 0.05 --confirm'

echo '== 3/5 FTS/vector indexes =='
echo 'Run after the FTS backfill completes:'
echo '  psql -h "$POSTGRES_HOST" -U "$POSTGRES_USER" -d "$POSTGRES_DATABASE" -f sql/011_v6_search_indexes.sql'

echo '== 5/5 Verify =='
echo '  python scripts/check_search_stack.py'
echo '  python scripts/verify_vectors.py'
