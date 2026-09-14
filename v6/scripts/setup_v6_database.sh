#!/usr/bin/env bash
set -euo pipefail

: "${POSTGRES_HOST:?POSTGRES_HOST is required}"
: "${POSTGRES_USER:?POSTGRES_USER is required}"
: "${POSTGRES_DATABASE:?POSTGRES_DATABASE is required}"

PSQL=(psql -h "$POSTGRES_HOST" -U "$POSTGRES_USER" -d "$POSTGRES_DATABASE")

echo '== 1/4 Foundation =='
"${PSQL[@]}" -f sql/010_v6_search_foundation.sql

echo '== 2/4 FTS backfill =='
echo 'Run the following separately after checking production load:'
echo '  python scripts/backfill_search_vector_v6.py --limit 1000 --batch-size 500 --sleep 0.20 --confirm'
echo '  python scripts/backfill_search_vector_v6.py --batch-size 1000 --sleep 0.05 --confirm'

echo '== 3/4 Indexes =='
echo 'Run after the FTS backfill completes:'
echo '  psql -h "$POSTGRES_HOST" -U "$POSTGRES_USER" -d "$POSTGRES_DATABASE" -f sql/011_v6_search_indexes.sql'

echo '== 4/4 Verify =='
echo '  python scripts/check_search_stack.py'
echo '  python scripts/verify_vectors.py'
