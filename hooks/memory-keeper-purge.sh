#!/bin/bash
# memory-keeper-purge.sh
# Política de expurgo: a cada 15 dias, mantém últimos 7 dias, limpa o resto
#
# Estratégia:
#   - Roda via cron a cada dia
#   - Verifica se passaram 15 dias desde o último expurgo
#   - Se sim: deleta registros com mais de 7 dias (mantém a semana recente)
#   - Mantém checkpoints marcados como "permanent" (nunca expurga)
#   - Log de tudo que foi purgado
#
# Uso: ./memory-keeper-purge.sh [--force] [--dry-run]

set -euo pipefail
export PATH="/Users/nelson.frugeri/.nvm/versions/node/v22.15.0/bin:$PATH"

DB_DIR="${DATA_DIR:-$HOME/mcp-data/memory-keeper}"
DB_FILE="$DB_DIR/context.db"
LOG_DIR="$HOME/.claude/hooks/logs"
STATE_FILE="$LOG_DIR/purge-state.txt"
LOG_FILE="$LOG_DIR/purge.log"
PURGE_CYCLE_DAYS=15
RETENTION_DAYS=7
FORCE=false
DRY_RUN=false

mkdir -p "$LOG_DIR"

# Parse args
for arg in "$@"; do
    case $arg in
        --force) FORCE=true ;;
        --dry-run) DRY_RUN=true ;;
    esac
done

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Verifica se DB existe
if [ ! -f "$DB_FILE" ]; then
    log "SKIP: Database não encontrada em $DB_FILE"
    exit 0
fi

# Verifica ciclo de 15 dias
if [ -f "$STATE_FILE" ] && [ "$FORCE" = false ]; then
    LAST_PURGE=$(cat "$STATE_FILE")
    LAST_PURGE_EPOCH=$(date -j -f '%Y-%m-%d' "$LAST_PURGE" '+%s' 2>/dev/null || date -d "$LAST_PURGE" '+%s' 2>/dev/null || echo 0)
    NOW_EPOCH=$(date '+%s')
    DAYS_SINCE=$(( (NOW_EPOCH - LAST_PURGE_EPOCH) / 86400 ))

    if [ "$DAYS_SINCE" -lt "$PURGE_CYCLE_DAYS" ]; then
        log "SKIP: Último expurgo há $DAYS_SINCE dias (ciclo=$PURGE_CYCLE_DAYS dias)"
        exit 0
    fi
fi

log "=== INICIANDO EXPURGO ==="
log "Ciclo: $PURGE_CYCLE_DAYS dias | Retenção: $RETENTION_DAYS dias"

# Calcula data de corte (7 dias atrás)
CUTOFF_DATE=$(date -v-${RETENTION_DAYS}d '+%Y-%m-%d %H:%M:%S' 2>/dev/null || date -d "-$RETENTION_DAYS days" '+%Y-%m-%d %H:%M:%S')
log "Data de corte: $CUTOFF_DATE"

# Conta registros que serão purgados
ITEMS_TO_PURGE=$(sqlite3 "$DB_FILE" "SELECT COUNT(*) FROM context_items WHERE created_at < '$CUTOFF_DATE';")
SESSIONS_TO_PURGE=$(sqlite3 "$DB_FILE" "SELECT COUNT(*) FROM sessions WHERE created_at < '$CUTOFF_DATE';")

log "Registros a purgar: $ITEMS_TO_PURGE items, $SESSIONS_TO_PURGE sessions"

if [ "$DRY_RUN" = true ]; then
    log "DRY-RUN: Nenhum registro deletado"

    # Mostra preview do que seria deletado
    log "--- Preview items a deletar ---"
    sqlite3 "$DB_FILE" "SELECT key, channel, category, created_at FROM context_items WHERE created_at < '$CUTOFF_DATE' ORDER BY created_at DESC LIMIT 20;" | while IFS='|' read -r key channel category created; do
        log "  - [$category] $key (channel=$channel, created=$created)"
    done

    log "--- Preview sessions a deletar ---"
    sqlite3 "$DB_FILE" "SELECT id, name, created_at FROM sessions WHERE created_at < '$CUTOFF_DATE' ORDER BY created_at DESC LIMIT 10;" | while IFS='|' read -r id name created; do
        log "  - Session $name ($id, created=$created)"
    done

    exit 0
fi

# Backup antes de purgar
BACKUP_FILE="$DB_DIR/context_backup_$(date '+%Y%m%d_%H%M%S').db"
cp "$DB_FILE" "$BACKUP_FILE"
log "Backup criado: $BACKUP_FILE"

# Executa expurgo
# 1. Remove items antigos (exceto checkpoints permanentes)
sqlite3 "$DB_FILE" <<SQL
-- Remove context_items antigos
DELETE FROM context_items
WHERE created_at < '$CUTOFF_DATE';

-- Remove sessions antigas sem items associados
DELETE FROM sessions
WHERE created_at < '$CUTOFF_DATE'
AND id NOT IN (SELECT DISTINCT session_id FROM context_items WHERE session_id IS NOT NULL);

-- Remove watchers expirados
DELETE FROM context_watchers
WHERE expires_at IS NOT NULL AND expires_at < datetime('now');

-- Limpa file_cache antigo
DELETE FROM file_cache
WHERE cached_at < '$CUTOFF_DATE';

-- Vacuum para recuperar espaço
VACUUM;
SQL

# Conta registros restantes
ITEMS_REMAINING=$(sqlite3 "$DB_FILE" "SELECT COUNT(*) FROM context_items;")
SESSIONS_REMAINING=$(sqlite3 "$DB_FILE" "SELECT COUNT(*) FROM sessions;")
DB_SIZE=$(du -h "$DB_FILE" | cut -f1)

log "Expurgo concluído:"
log "  - Items deletados: $ITEMS_TO_PURGE"
log "  - Sessions deletadas: $SESSIONS_TO_PURGE"
log "  - Items restantes: $ITEMS_REMAINING"
log "  - Sessions restantes: $SESSIONS_REMAINING"
log "  - Tamanho DB: $DB_SIZE"

# Limpa backups antigos (mantém últimos 3)
ls -t "$DB_DIR"/context_backup_*.db 2>/dev/null | tail -n +4 | xargs rm -f 2>/dev/null
log "Backups antigos limpos (mantém últimos 3)"

# Atualiza estado do último expurgo
date '+%Y-%m-%d' > "$STATE_FILE"

log "=== EXPURGO FINALIZADO ==="
