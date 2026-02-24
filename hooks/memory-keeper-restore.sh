#!/bin/bash
# memory-keeper-restore.sh
# Auto-inject context restoration reminder on session start
# Triggered by: SessionStart hook

INPUT=$(cat)
SOURCE=$(echo "$INPUT" | jq -r '.source // "startup"')
CWD=$(echo "$INPUT" | jq -r '.cwd // "unknown"')
PROJECT_NAME=$(basename "$CWD")
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

# Log
LOG_DIR="$HOME/.claude/hooks/logs"
mkdir -p "$LOG_DIR"
echo "[$TIMESTAMP] SessionStart source=$SOURCE Project=$PROJECT_NAME" >> "$LOG_DIR/memory-keeper.log"

# Injecta instrução para restaurar contexto
cat <<EOF
[MEMORY-KEEPER] Sessão iniciada (source: $SOURCE) para projeto "$PROJECT_NAME".
IMPORTANTE: Use context_get com channel "$PROJECT_NAME" para restaurar o contexto da última sessão.
Se for a primeira vez neste projeto, use context_session_start com projectDir="$CWD".
EOF

exit 0
