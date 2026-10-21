#!/bin/bash
# Serve the generated dashboard over the tailnet from fokker: bash scripts/serve_dashboard.sh
# Idempotent: when something already listens on the address it only prints the URLs.
# The server is a plain http.server in the background (nohup); its pid and log live outside the served
# directory in .scratch/dashboard_state/. Stop it with: kill $(cat .scratch/dashboard_state/server.pid)
set -euo pipefail
PROJECT=/mnt/data0/shared/colin/Projects/SchrodingerDistillation
DIRECTORY="$PROJECT/dashboard"
STATE="$PROJECT/.scratch/dashboard_state"
ADDRESS=100.92.190.10
PORT=8892
mkdir -p "$DIRECTORY" "$STATE"
if [ -n "$(ss -ltnH "sport = :$PORT")" ]; then
  echo "already listening on port $PORT (pid $(cat "$STATE/server.pid" 2>/dev/null || echo unknown))"
else
  nohup python3 -m http.server "$PORT" --bind "$ADDRESS" --directory "$DIRECTORY" >> "$STATE/server.log" 2>&1 &
  echo $! > "$STATE/server.pid"
  sleep 1
  [ -n "$(ss -ltnH "sport = :$PORT")" ] || { echo "server did not start; see $STATE/server.log"; exit 1; }
  echo "serving $DIRECTORY (pid $(cat "$STATE/server.pid"))"
fi
echo "http://$ADDRESS:$PORT/"
echo "http://fokker.tail02e69.ts.net:$PORT/"
