#!/bin/bash
# setup-cron.sh
# Execute este script FORA do Claude Code para instalar o cron job
# Uso: bash ~/.claude/hooks/setup-cron.sh

CRON_CMD="0 3 * * * $HOME/.claude/hooks/memory-keeper-purge.sh >> $HOME/.claude/hooks/logs/purge-cron.log 2>&1"

# Verifica se já existe
if crontab -l 2>/dev/null | grep -q "memory-keeper-purge"; then
    echo "Cron job já existe. Nenhuma alteração feita."
    crontab -l | grep "memory-keeper-purge"
else
    (crontab -l 2>/dev/null; echo "$CRON_CMD") | crontab -
    echo "Cron job instalado com sucesso!"
    echo "Verificação: roda diariamente às 3h, expurga a cada 15 dias (mantém 7 dias)"
    crontab -l | grep "memory-keeper-purge"
fi
