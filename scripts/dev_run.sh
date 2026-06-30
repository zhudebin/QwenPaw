#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# QwenPaw — local source-based dev launcher (uv-powered)
#
# What it does:
#   1. Make sure ``uv`` is installed (auto-installs if missing).
#   2. Resolve / create the working directory; export it as
#      ``QWENPAW_WORKING_DIR`` so the app uses it.
#   3. Load ``.env`` (if present) so secrets (API keys, etc.) are available.
#   4. ``uv sync`` to materialize the venv at ``.venv`` (uses the vendored
#      ``vendor/agentscope`` per pyproject.toml).
#   5. ``qwenpaw init`` (idempotent) the first time the working dir is empty.
#   6. ``qwenpaw app --host <HOST> --port <PORT>`` — that's it.
#
# Quick start:
#   ./scripts/dev_run.sh                 # backend only (~/.qwenpaw-dev, :8088)
#   ./scripts/dev_run.sh --with-frontend # backend + Vite dev server on :5173
#   QWENPAW_WORKING_DIR=/tmp/qp ./scripts/dev_run.sh
#   QWENPAW_PORT=9000 ./scripts/dev_run.sh
#   ./scripts/dev_run.sh --build-console # also (re)build the React console
#   ./scripts/dev_run.sh --skip-sync     # skip uv sync (faster restart)
#
# All env vars honoured:
#   QWENPAW_WORKING_DIR  Working dir (default: ~/.qwenpaw-dev)
#   QWENPAW_PORT         HTTP port (default: 8088)
#   QWENPAW_HOST         Bind address (default: 127.0.0.1)
#   QWENPAW_FRONTEND_PORT Vite dev port when --with-frontend (default: 5173)
#   QWENPAW_AUTH_*       Auth knobs forwarded as-is.
# ---------------------------------------------------------------------------
set -euo pipefail

# ---- Paths -----------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# ---- Pretty logging --------------------------------------------------------
if [[ -t 1 ]]; then
  C_DIM=$'\033[2m'; C_BLU=$'\033[34m'; C_GRN=$'\033[32m'
  C_YLW=$'\033[33m'; C_RED=$'\033[31m'; C_RST=$'\033[0m'
else
  C_DIM=""; C_BLU=""; C_GRN=""; C_YLW=""; C_RED=""; C_RST=""
fi
log()  { printf '%s[dev_run]%s %s\n'  "${C_BLU}" "${C_RST}" "$*"; }
ok()   { printf '%s[dev_run]%s %s\n'  "${C_GRN}" "${C_RST}" "$*"; }
warn() { printf '%s[dev_run]%s %s\n'  "${C_YLW}" "${C_RST}" "$*" >&2; }
die()  { printf '%s[dev_run]%s %s\n'  "${C_RED}" "${C_RST}" "$*" >&2; exit 1; }

# ---- Argument parsing ------------------------------------------------------
BUILD_CONSOLE=0
SKIP_SYNC=0
WITH_FRONTEND=0
EXTRA_ARGS=()

usage() {
  sed -n '2,30p' "$0"
  exit 0
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --build-console)  BUILD_CONSOLE=1;  shift ;;
    --skip-sync)      SKIP_SYNC=1;      shift ;;
    --with-frontend)  WITH_FRONTEND=1;  shift ;;
    --no-frontend)    WITH_FRONTEND=0;  shift ;;
    -h|--help)        usage ;;
    --)               shift; EXTRA_ARGS+=("$@"); break ;;
    *)                EXTRA_ARGS+=("$1"); shift ;;
  esac
done

# ---- Load .env (project root) ---------------------------------------------
if [[ -f "${PROJECT_ROOT}/.env" ]]; then
  log "Loading .env from project root"
  # shellcheck disable=SC1091
  set -a; . "${PROJECT_ROOT}/.env"; set +a
fi

# ---- Working directory (extracted to env var) -----------------------------
: "${QWENPAW_WORKING_DIR:=${HOME}/.qwenpaw-dev}"
mkdir -p "${QWENPAW_WORKING_DIR}"
export QWENPAW_WORKING_DIR
ok "QWENPAW_WORKING_DIR = ${QWENPAW_WORKING_DIR}"

: "${QWENPAW_HOST:=127.0.0.1}"
: "${QWENPAW_PORT:=8088}"
: "${QWENPAW_FRONTEND_PORT:=5173}"
export QWENPAW_HOST QWENPAW_PORT QWENPAW_FRONTEND_PORT

# ---- Ensure uv is on PATH --------------------------------------------------
ensure_uv() {
  if command -v uv >/dev/null 2>&1; then
    return
  fi
  warn "uv not found; installing via official installer..."
  if command -v curl >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
  elif command -v wget >/dev/null 2>&1; then
    wget -qO- https://astral.sh/uv/install.sh | sh
  else
    die "Need either curl or wget to install uv. Aborting."
  fi
  # shellcheck disable=SC1091
  [[ -f "${HOME}/.local/bin/env" ]] && . "${HOME}/.local/bin/env"
  export PATH="${HOME}/.local/bin:${PATH}"
  command -v uv >/dev/null 2>&1 || die "uv install failed"
}
ensure_uv
log "uv: $(uv --version)"

# ---- Vendor sanity check ---------------------------------------------------
if [[ ! -f "${PROJECT_ROOT}/vendor/agentscope/pyproject.toml" ]]; then
  die "vendor/agentscope is missing. Restore it before running."
fi

# ---- Sync deps -------------------------------------------------------------
if [[ "${SKIP_SYNC}" == "1" ]]; then
  warn "--skip-sync: not running 'uv sync'"
else
  log "Running 'uv sync' (this may take a while on first run)"
  # Always reinstall vendored agentscope: it's resolved from a local path
  # (vendor/agentscope), and uv won't pick up edits to its source files
  # without an explicit --reinstall-package.
  uv sync --reinstall-package agentscope
  ok "Dependencies in sync"
fi

# ---- Optional console build ------------------------------------------------
if [[ "${BUILD_CONSOLE}" == "1" ]]; then
  if ! command -v npm >/dev/null 2>&1; then
    die "--build-console requested but 'npm' not found"
  fi
  log "Building console (npm ci && npm run build)"
  (
    cd "${PROJECT_ROOT}/console"
    [[ -d node_modules ]] || npm ci
    npm run build
  )
  log "Copying console/dist -> src/qwenpaw/console/"
  rm -rf "${PROJECT_ROOT}/src/qwenpaw/console"
  mkdir -p "${PROJECT_ROOT}/src/qwenpaw/console"
  cp -R "${PROJECT_ROOT}/console/dist/." "${PROJECT_ROOT}/src/qwenpaw/console/"
  ok "Console built"
elif [[ ! -d "${PROJECT_ROOT}/src/qwenpaw/console" ]]; then
  warn "src/qwenpaw/console/ is empty — UI will 404. Run with --build-console to build it."
fi

# ---- First-run init --------------------------------------------------------
if [[ ! -f "${QWENPAW_WORKING_DIR}/config.json" ]]; then
  log "config.json missing; running 'qwenpaw init --defaults --accept-security'"
  uv run qwenpaw init --defaults --accept-security
  ok "Init complete"
else
  ok "config.json found, skipping init"
fi

# ---- Optional frontend dev server (Vite) ----------------------------------
FRONTEND_PID=""

cleanup() {
  if [[ -n "${FRONTEND_PID}" ]] && kill -0 "${FRONTEND_PID}" 2>/dev/null; then
    log "Stopping frontend dev server (pid=${FRONTEND_PID})"
    # Kill the whole process group so child node processes die too.
    kill -TERM -- "-${FRONTEND_PID}" 2>/dev/null || kill -TERM "${FRONTEND_PID}" 2>/dev/null || true
    wait "${FRONTEND_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

if [[ "${WITH_FRONTEND}" == "1" ]]; then
  if ! command -v npm >/dev/null 2>&1; then
    die "--with-frontend requires 'npm' on PATH"
  fi
  if [[ "${BUILD_CONSOLE}" == "1" ]]; then
    warn "--with-frontend implies dev server (Vite); ignoring --build-console for production build"
  fi
  CONSOLE_DIR="${PROJECT_ROOT}/console"
  if [[ ! -d "${CONSOLE_DIR}/node_modules" ]]; then
    log "console/node_modules missing; running 'npm ci'"
    ( cd "${CONSOLE_DIR}" && npm ci )
  fi
  log "Launching Vite dev server on :${QWENPAW_FRONTEND_PORT} (proxy -> ${QWENPAW_HOST}:${QWENPAW_PORT})"
  # Run npm in its own process group so cleanup() can kill the whole tree.
  (
    cd "${CONSOLE_DIR}"
    QWENPAW_BACKEND="${QWENPAW_HOST}:${QWENPAW_PORT}" \
    exec npm run dev -- --port "${QWENPAW_FRONTEND_PORT}"
  ) &
  FRONTEND_PID=$!
  ok "Vite dev server PID=${FRONTEND_PID} (logs interleaved below)"
fi

# ---- Launch backend (foreground) ------------------------------------------
log "Starting QwenPaw backend on ${QWENPAW_HOST}:${QWENPAW_PORT}"
if [[ "${WITH_FRONTEND}" == "1" ]]; then
  log "Open the UI at: http://${QWENPAW_HOST}:${QWENPAW_FRONTEND_PORT}"
fi
log "Logs are streamed to this terminal. Ctrl+C to stop everything."
printf '%s%s%s\n' "${C_DIM}" "----------------------------------------------" "${C_RST}"
# Note: ``${arr[@]:+...}`` is the bash 3.2-safe way to expand a possibly-empty
# array under ``set -u``. We avoid `exec` so the EXIT trap can clean up the
# Vite child process when the backend stops.
uv run qwenpaw app \
  --host "${QWENPAW_HOST}" \
  --port "${QWENPAW_PORT}" \
  ${EXTRA_ARGS[@]:+"${EXTRA_ARGS[@]}"}
