#!/bin/bash
# memory-keeper-save.sh
# Auto-save context to Memory Keeper before session ends or compaction
# Triggered by: PreCompact, Stop hooks

INPUT=$(cat)
SESSION_ID=$(echo "$INPUT" | jq -r '.session_id // "unknown"')
CWD=$(echo "$INPUT" | jq -r '.cwd // "unknown"')
EVENT=$(echo "$INPUT" | jq -r '.hook_event_name // "unknown"')
PROJECT_NAME=$(basename "$CWD")
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

# Log para debug
LOG_DIR="$HOME/.claude/hooks/logs"
mkdir -p "$LOG_DIR"
echo "[$TIMESTAMP] Event=$EVENT Session=$SESSION_ID Project=$PROJECT_NAME CWD=$CWD" >> "$LOG_DIR/memory-keeper.log"

# Para PreCompact, injecta lembrete no contexto
if [ "$EVENT" = "PreCompact" ]; then
    cat <<EOF
[MEMORY-KEEPER] Antes da compactação, salve o contexto importante usando memory-keeper:
- Use context_save para salvar decisões, progresso e próximos passos do projeto "$PROJECT_NAME"
- Use context_checkpoint para criar um snapshot completo
- Channel sugerido: "$PROJECT_NAME"
EOF
fi

exit 0
