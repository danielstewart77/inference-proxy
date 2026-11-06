#!/usr/bin/env bash
# Put the proxy on Postgres and prove which database answered.
#
# The first attempt reported success while the proxy was still on its SQLite
# file, because /health and the model listing read the same from either store.
# The only honest proof is a write: make a real request and see which store
# the usage row lands in.
set -uo pipefail
PROXY=/home/daniel/Storage/Dev/inference-proxy
REPORT="$PROXY/.cutover/switch-report.txt"
DSN=$(grep '^PROXY_TEST_DSN=' "$PROXY/.env" | cut -d= -f2-)
KEY=$(grep -o 'hmp-[A-Za-z0-9_-]*' /home/daniel/Storage/hive-edge-mind-skippy/.env | head -1)

exec > >(tee -a "$REPORT") 2>&1
echo "=== switch started $(date -Is)"

pg_count() { docker exec -i hive-postgres psql -U hive -d hive -tAc \
  "select count(*) from proxy.usage_log"; }
sq_count() { sqlite3 "$PROXY/data/proxy.db" "select count(*) from usage_log"; }

start_on() {  # $1 = "postgres" | "sqlite"
  cd "$PROXY" || exit 1
  if [ "$1" = "postgres" ]; then
    DATABASE_URL="$DSN" DB_SCHEMA=proxy setsid nohup \
      "$PROXY/.venv/bin/python" -m app.entrypoint \
      >> "$PROXY/.cutover/proxy-stdout.log" 2>&1 < /dev/null &
  else
    env -u DATABASE_URL -u DB_SCHEMA setsid nohup \
      "$PROXY/.venv/bin/python" -m app.entrypoint \
      >> "$PROXY/.cutover/proxy-stdout.log" 2>&1 < /dev/null &
  fi
  disown
}

stop_all() {
  # By port, not by command line: a listener this script did not recognise is
  # exactly how the last attempt ended up verifying a process it had not
  # started.
  fuser -k -TERM 8899/tcp 2>/dev/null
  sleep 3
  fuser -k -KILL 8899/tcp 2>/dev/null
  sleep 2
  if fuser 8899/tcp 2>/dev/null; then echo "WARNING: 8899 still held"; fi
}

healthy() {
  for _ in $(seq 1 45); do
    curl -fsS --max-time 3 http://127.0.0.1:8899/health > /dev/null 2>&1 && return 0
    sleep 1
  done
  return 1
}

rollback() {
  echo "---- ROLLING BACK to the SQLite file ----"
  sed -i '/^DATABASE_URL=/d;/^DB_SCHEMA=/d' "$PROXY/.env"
  stop_all
  start_on sqlite
  healthy && echo "rolled back; proxy healthy on SQLite" \
          || echo "ROLLBACK FAILED -- start by hand: cd $PROXY && .venv/bin/python -m app.entrypoint"
  echo "=== switch ABORTED $(date -Is)"
  exit 1
}

echo "-- before: postgres=$(pg_count) sqlite=$(sq_count)"

echo "-- stopping everything on 8899"
stop_all

echo "-- starting on Postgres, with the location stated in the environment"
start_on postgres
healthy || rollback
echo "health: $(curl -fsS --max-time 5 http://127.0.0.1:8899/health)"

echo "-- which database is it actually connected to"
PID=$(fuser 8899/tcp 2>/dev/null | tr -d ' ')
echo "pid: $PID"
echo "open sqlite handles: $(ls -l /proc/$PID/fd 2>/dev/null | grep -c 'data/proxy.db')"
echo "connections to 5433: $(ss -tn 2>/dev/null | grep -c ':5433')"

echo "-- a real request, and where its usage row lands"
PG0=$(pg_count); SQ0=$(sq_count)
CODE=$(curl -s -o /tmp/sw.json -w '%{http_code}' --max-time 60 \
  -H "Authorization: Bearer $KEY" -H "content-type: application/json" \
  -d '{"model":"claude-sonnet-5","max_tokens":16,"messages":[{"role":"user","content":"say the single word: cutover"}]}' \
  http://127.0.0.1:8899/v1/messages)
echo "POST /v1/messages -> $CODE"
head -c 300 /tmp/sw.json; echo
sleep 6
PG1=$(pg_count); SQ1=$(sq_count)
echo "postgres usage_log: $PG0 -> $PG1"
echo "sqlite   usage_log: $SQ0 -> $SQ1"

if [ "$CODE" != "200" ]; then echo "the proxy refused a real inference request"; rollback; fi
if [ "$PG1" -le "$PG0" ]; then
  echo "the usage row did NOT land in Postgres -- the proxy is not reading the"
  echo "one database no matter what it reports"
  rollback
fi
if [ "$SQ1" -ne "$SQ0" ]; then
  echo "the SQLite file is still being written -- something else is serving"
  rollback
fi

echo "-- model listing still matches"
echo "models: $(curl -fsS --max-time 10 -H "Authorization: Bearer $KEY" \
  http://127.0.0.1:8899/v1/models | python3 -c "
import json,sys; print(len(json.load(sys.stdin).get('data', [])))")"

echo "=== switch COMPLETE $(date -Is): proxy is on Postgres and proved it by a write"
