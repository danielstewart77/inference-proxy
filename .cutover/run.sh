#!/usr/bin/env bash
# The proxy cutover, as one unattended run.
#
# It has to be unattended: every mind on this host reaches its model through
# this proxy, so while it is down nothing here can think -- including whatever
# is running this. So every decision is made in advance, and every failure
# path ends with the proxy back on its SQLite file rather than with a prompt
# nobody can answer.
set -uo pipefail

PROXY=/home/daniel/Storage/Dev/inference-proxy
HIVE=/home/daniel/Storage/Dev/hive_mind
REPORT="$PROXY/.cutover/report.txt"
DSN=$(grep '^PROXY_TEST_DSN=' "$PROXY/.env" | cut -d= -f2-)

exec > >(tee -a "$REPORT") 2>&1
echo "=============================================================="
echo "cutover started $(date -Is)"
echo "=============================================================="

say() { echo; echo "---- $* ----"; }

start_proxy() {
  cd "$PROXY" || exit 1
  setsid nohup "$PROXY/.venv/bin/python" -m app.entrypoint \
    >> "$PROXY/proxy.log" 2>&1 < /dev/null &
  disown
}

wait_healthy() {
  for _ in $(seq 1 45); do
    if curl -fsS --max-time 3 http://127.0.0.1:8899/health > /dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  return 1
}

rollback() {
  say "ROLLING BACK to the SQLite file"
  sed -i '/^DATABASE_URL=/d;/^DB_SCHEMA=/d' "$PROXY/.env"
  pkill -f "$PROXY/.venv/bin/python -m app.entrypoint"
  sleep 2
  start_proxy
  if wait_healthy; then
    echo "ROLLED BACK: proxy healthy on SQLite again"
  else
    echo "ROLLBACK FAILED: proxy is NOT answering. Manual start needed:"
    echo "  cd $PROXY && .venv/bin/python -m app.entrypoint"
  fi
  echo "cutover ABORTED $(date -Is)"
  exit 1
}

say "0. before: what the proxy answers now, to compare against afterwards"
KEY=$(grep -o 'hmp-[A-Za-z0-9_-]*' /home/daniel/Storage/hive-edge-mind-skippy/.env 2>/dev/null | head -1)
BEFORE_MODELS=$(curl -fsS --max-time 10 -H "Authorization: Bearer $KEY" \
  http://127.0.0.1:8899/v1/models | python3 -c "
import json,sys; print(len(json.load(sys.stdin).get('data', [])))" 2>/dev/null)
echo "GET /v1/models now returns $BEFORE_MODELS models"
if [ -z "$BEFORE_MODELS" ] || [ "$BEFORE_MODELS" = "0" ]; then
  echo "REFUSING: cannot read a baseline from the running proxy, so there is"
  echo "nothing to verify the cutover against. Nothing has been changed."
  exit 2
fi

say "1. before: what SQLite holds"
sqlite3 "$PROXY/data/proxy.db" \
  "select 'clients',count(*) from clients union all select 'keys',count(*) from keys
   union all select 'credentials',count(*) from credentials
   union all select 'models',count(*) from models
   union all select 'providers',count(*) from providers
   union all select 'usage_log',count(*) from usage_log;"

say "2. stopping the proxy -- every mind on this host is mute from here"
pkill -f "$PROXY/.venv/bin/python -m app.entrypoint"
for _ in $(seq 1 20); do pgrep -f "$PROXY/.venv/bin/python -m app.entrypoint" > /dev/null || break; sleep 1; done
if pgrep -f "$PROXY/.venv/bin/python -m app.entrypoint" > /dev/null; then
  echo "proxy did not stop; killing"
  pkill -9 -f "$PROXY/.venv/bin/python -m app.entrypoint"; sleep 2
fi
echo "stopped at $(date -Is)"

say "3. bringing the proxy schema to head BEFORE the load"
# The loader refuses when the source carries a column the target lacks, and
# correctly: it would drop the column and then call the table a match.
cd "$PROXY" || rollback
DATABASE_URL="$DSN" DB_SCHEMA=proxy "$PROXY/.venv/bin/python" -c \
  "from app.entrypoint import bootstrap_schema; bootstrap_schema()" || rollback

say "4. settling the SQLite file before anything reads it"
# A killed writer leaves a write-ahead log behind, and a reader that cannot
# play it back sees a different number of rows than one that can. Folding it
# into the file first means the snapshot and the re-count are reading the
# same thing, so any drift the loader reports is real movement rather than an
# artifact of how it was read.
sqlite3 "$PROXY/data/proxy.db" "PRAGMA wal_checkpoint(TRUNCATE);" || rollback
BRACKET_BEFORE=$(sqlite3 "$PROXY/data/proxy.db" "select count(*) from usage_log")
echo "usage_log before the load: $BRACKET_BEFORE"
ls -la "$PROXY/data/proxy.db"* | grep -E "wal|shm" || echo "no write-ahead log left"

say "5. loading the live data, proxy only -- twice"
# Twice on purpose. The first pass is a rehearsal whose only job is to leave a
# snapshot and a drift report; the second has to come back with nothing moved.
# A single pass cannot tell "the writer is stopped" from "the writer was
# stopped a moment ago and something was still finishing", and the difference
# decides whether the database about to become the source of truth is a copy
# of the file or a copy of a moment in the file's past.
cd "$HIVE" || rollback
for PASS in 1 2; do
  echo
  echo ">>> load pass $PASS"
  "$HIVE/.venv/bin/python" -m unified_db.load_data --only proxy --dsn "$DSN"
  LOAD=$?
  echo "pass $PASS exit: $LOAD"
  [ "$LOAD" -eq 0 ] && break
done

BRACKET_AFTER=$(sqlite3 "$PROXY/data/proxy.db" "select count(*) from usage_log")
echo "usage_log after the load:  $BRACKET_AFTER"

if [ "$LOAD" -ne 0 ]; then
  echo
  echo "the second pass still reports movement in a file whose only writer is"
  echo "stopped. Not switching. What moved, so it can be read with the proxy"
  echo "up again:"
  SNAP=$(grep -o '/tmp/hive-load-[a-z0-9]*' "$REPORT" | tail -1)
  if [ -n "$SNAP" ] && [ -f "$SNAP/proxy.db" ]; then
    echo "-- in the snapshot, gone from the live file (oldest first):"
    sqlite3 "$SNAP/proxy.db" \
      "attach '$PROXY/data/proxy.db' as live;
       select id, created_at, model_name from main.usage_log
       where id not in (select id from live.usage_log)
       order by created_at limit 5;"
    echo "-- in the live file, absent from the snapshot:"
    sqlite3 "$SNAP/proxy.db" \
      "attach '$PROXY/data/proxy.db' as live;
       select id, created_at, model_name from live.usage_log
       where id not in (select id from main.usage_log)
       order by created_at limit 5;"
  fi
  echo "-- anything still holding the file open:"
  for pid in $(pgrep -f "inference-proxy"); do
    echo "   pid $pid: $(tr '\0' ' ' < /proc/$pid/cmdline 2>/dev/null | cut -c1-120)"
  done
  rollback
fi

say "6. pointing the proxy at the one database"
sed -i '/^DATABASE_URL=/d;/^DB_SCHEMA=/d' "$PROXY/.env"
{
  echo ""
  echo "# The one database. Unset these two lines and restart to go back to"
  echo "# the SQLite file, which is left in place."
  echo "DATABASE_URL=$DSN"
  echo "DB_SCHEMA=proxy"
} >> "$PROXY/.env"

say "7. starting the proxy on Postgres"
start_proxy
wait_healthy || rollback
echo "health: $(curl -fsS --max-time 5 http://127.0.0.1:8899/health)"

say "8. does it answer a real request, with a real key, as before"
CODE=$(curl -s -o /tmp/models.json -w '%{http_code}' --max-time 10 \
  -H "Authorization: Bearer $KEY" http://127.0.0.1:8899/v1/models)
AFTER_MODELS=$(python3 -c "
import json
try:
    print(len(json.load(open('/tmp/models.json')).get('data', [])))
except Exception:
    print(0)")
echo "GET /v1/models -> $CODE, $AFTER_MODELS models (was $BEFORE_MODELS)"
if [ "$CODE" != "200" ]; then
  echo "the proxy is up but refused a real request"
  rollback
fi
if [ "$AFTER_MODELS" != "$BEFORE_MODELS" ]; then
  echo "the model listing changed across the cutover: $BEFORE_MODELS -> $AFTER_MODELS"
  rollback
fi

say "9. what Postgres holds now, against the source"
docker exec -i hive-postgres psql -U hive -d hive -tAc \
  "select 'clients',count(*) from proxy.clients union all select 'keys',count(*) from proxy.keys
   union all select 'credentials',count(*) from proxy.credentials
   union all select 'models',count(*) from proxy.models
   union all select 'providers',count(*) from proxy.providers
   union all select 'usage_log',count(*) from proxy.usage_log;"
echo "-- identity counters:"
docker exec -i hive-postgres psql -U hive -d hive -tAc \
  "select c.relname, s.last_value from pg_class c
   join pg_sequences s on s.sequencename = c.relname
   where c.relnamespace = 'proxy'::regnamespace and c.relkind = 'S' order by 1;"
echo "-- schema revision:"
docker exec -i hive-postgres psql -U hive -d hive -tAc "select * from proxy.alembic_version;"

say "10. nothing leaked into the shared schema"
docker exec -i hive-postgres psql -U hive -d hive -tAc \
  "select count(*) from information_schema.tables where table_schema='public';"

echo
echo "cutover COMPLETE $(date -Is)"
