# Helpers for starting a server in its own session and stopping exactly that
# session (the server and every worker it spawned), without touching other
# processes on the machine.  Source from bash:
#
#   source scripts/server_lifecycle.sh
#   port_in_use "$PORT" && { echo "port busy"; exit 1; }
#   setsid vllm serve ... > server.log 2>&1 &   # in a non-interactive script,
#   SPID=$!                                     # $! is then the session leader
#   ...
#   stop_session "$SPID"; wait_port_free "$PORT"

# Descendants of $1 (recursively).
_descendants() {
  local c
  for c in $(pgrep -P "$1" 2>/dev/null); do echo "$c"; _descendants "$c"; done
}

# Every live process in session $1, plus descendants of $1 that left the session.
_session_pids() {
  { ps -eo pid=,sid= | awk -v s="$1" '$2 == s { print $1 }'; _descendants "$1"; } | sort -u
}

# stop_session <session leader pid>: SIGTERM the session, wait up to 30 s, then
# SIGKILL whatever is left of it. The session is re-scanned every second, so
# processes that join it during shutdown are signalled too (a session id cannot
# be reused while any member is alive).
stop_session() {
  local leader="${1:-}" pids new i
  [ -n "$leader" ] || return 0
  pids=$(_session_pids "$leader")
  [ -n "$pids" ] || return 0
  echo "  stopping server session $leader (up to 30 s)"
  kill -TERM $pids 2>/dev/null
  for i in $(seq 1 30); do
    sleep 1
    new=$(_session_pids "$leader")
    [ -z "$new" ] && return 0
    kill -TERM $(comm -13 <(printf '%s\n' $pids) <(printf '%s\n' $new)) 2>/dev/null
    pids="$new"
  done
  for i in 1 2 3 4 5; do  # SIGKILL until the session is empty (catches late forks)
    new=$(_session_pids "$leader")
    [ -z "$new" ] && return 0
    kill -KILL $new 2>/dev/null
    sleep 1
  done
}

# port_in_use <port>: true if something accepts connections on localhost:<port>.
port_in_use() {
  (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null
}

# wait_port_free <port> [seconds]: wait until the port is released (default 60 s).
wait_port_free() {
  local i
  for i in $(seq 1 "${2:-60}"); do port_in_use "$1" || return 0; sleep 1; done
  return 1
}

# wait_healthy <server pid> <port> [tries=300] [interval_s=4]: poll /health.
# Returns 0 once it answers 200, 1 if the server process exits, 2 after
# tries*interval seconds, 3 if the server is up but answers 5xx for 60 s
# (e.g. a broken dependency; a starting server refuses connections instead).
wait_healthy() {
  local pid="$1" port="$2" tries="${3:-300}" every="${4:-4}" i code n5=0
  for i in $(seq 1 "$tries"); do
    code=$(curl -s -o /dev/null -w '%{http_code}' "http://localhost:$port/health" 2>/dev/null)
    [ "$code" = 200 ] && return 0
    kill -0 "$pid" 2>/dev/null || return 1
    case "$code" in
      5??) n5=$((n5 + 1)); [ "$n5" -ge $((60 / every)) ] && return 3 ;;
      *) n5=0 ;;
    esac
    sleep "$every"
  done
  return 2
}

# health_failure <status of wait_healthy>: human-readable reason.
health_failure() {
  case "$1" in
    1) echo "server exited during startup" ;;
    2) echo "server not healthy before the startup timeout" ;;
    3) echo "server is up but /health keeps returning 5xx" ;;
    *) echo "server not healthy" ;;
  esac
}

# server_alive <server pid> <port>: true if the server process runs and /health answers 200.
server_alive() {
  kill -0 "$1" 2>/dev/null && [ "$(curl -s -o /dev/null -w '%{http_code}' "http://localhost:$2/health" 2>/dev/null)" = 200 ]
}

# Server-log lines that mean a worker died even though the API process may still
# answer /health (none of them occurs in the 113 server logs of the paper's runs).
FATAL_LOG_PATTERN='runtime_v2 entrypoint poll failed|pipe forwarder failed|CUDA out of memory|OutOfMemoryError|NCCL error|EngineDeadError|died unexpectedly'

# run_broken_reason <server pid> <port> <server log> <result json>: after a benchmark,
# print why the run must count as failed and return 0; return 1 if the run is fine.
run_broken_reason() {
  local pid="$1" port="$2" slog="$3" res="$4" m
  if ! server_alive "$pid" "$port"; then echo "server died or became unhealthy during the benchmark"; return 0; fi
  m=$(grep -m1 -oE "$FATAL_LOG_PATTERN" "$slog" 2>/dev/null)
  if [ -n "$m" ]; then echo "worker failure in the server log ('$m')"; return 0; fi
  if python3 -c 'import json,sys; sys.exit(json.load(open(sys.argv[1])).get("completed_requests", 1) != 0)' "$res" 2>/dev/null; then
    echo "no request completed"; return 0
  fi
  return 1
}
