#!/bin/bash
# bootstrap.sh
# Configura o ambiente .claude na máquina local
# Instala MCPs declarados no manifest, detecta Node, cria wrappers se necessário
#
# Cenários suportados:
#   A) Máquina nova (sem ~/.claude)     → clona o repo
#   B) Máquina com ~/.claude existente  → inicializa git e puxa do remote
#   C) Já bootstrapped                  → atualiza (git pull + reinstala MCPs)
#
# Uso:
#   Primeira vez (máquina nova):
#     REPO_URL=git@github.com:user/dotclaude.git bash <(curl -sL <raw-url>/bootstrap.sh)
#
#   Já tem o repo clonado:
#     bash ~/.claude/setup/bootstrap.sh          # instala MCPs
#     bash ~/.claude/setup/bootstrap.sh --check  # só verifica
#     bash ~/.claude/setup/bootstrap.sh --clean  # remove MCPs do bootstrap
#     bash ~/.claude/setup/bootstrap.sh --init   # clona/sincroniza repo

set -euo pipefail

CLAUDE_DIR="$HOME/.claude"
SETUP_DIR="$CLAUDE_DIR/setup"
HOOKS_DIR="$CLAUDE_DIR/hooks"
MANIFEST="$SETUP_DIR/mcp-manifest.json"
LOCAL_STATE="$SETUP_DIR/.local-state.json"
LOG_FILE="$SETUP_DIR/bootstrap.log"
REPO_URL="${REPO_URL:-}"
CHECK_ONLY=false
CLEAN=false
INIT=false

for arg in "$@"; do
    case $arg in
        --check) CHECK_ONLY=true ;;
        --clean) CLEAN=true ;;
        --init)  INIT=true ;;
    esac
done

log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $1"
    echo "$msg"
    echo "$msg" >> "$LOG_FILE"
}

error() {
    log "ERROR: $1"
    exit 1
}

# --- Step 0: Git Init/Sync ---

init_repo() {
    if [ -z "$REPO_URL" ]; then
        error "REPO_URL não definida. Uso: REPO_URL=git@github.com:user/dotclaude.git bash bootstrap.sh --init"
    fi

    if [ ! -d "$CLAUDE_DIR" ]; then
        # Cenário A: máquina nova, sem ~/.claude
        echo "[init] Clonando repo em $CLAUDE_DIR..."
        git clone "$REPO_URL" "$CLAUDE_DIR"
        echo "[init] ✓ Clone concluído"
        return
    fi

    if [ -d "$CLAUDE_DIR/.git" ]; then
        # Cenário C: já é um repo git
        echo "[init] ~/.claude já é um repo git. Atualizando..."
        cd "$CLAUDE_DIR"
        git fetch origin
        git pull --rebase origin main || git pull --rebase origin master
        echo "[init] ✓ Atualizado"
        return
    fi

    # Cenário B: ~/.claude existe mas NÃO é um repo git
    # Estratégia: init, add remote, fetch, merge preservando arquivos locais
    echo "[init] ~/.claude existe mas não é um repo git"
    echo "[init] Fazendo backup dos arquivos locais e sincronizando com o repo..."

    cd "$CLAUDE_DIR"

    # Backup de arquivos que podem conflitar
    BACKUP_DIR="$HOME/.claude-backup-$(date '+%Y%m%d_%H%M%S')"
    mkdir -p "$BACKUP_DIR"

    # Salva lista de arquivos locais existentes
    find . -maxdepth 1 -not -name '.' -not -name '..' | while read -r item; do
        cp -a "$item" "$BACKUP_DIR/" 2>/dev/null || true
    done
    echo "[init] Backup salvo em $BACKUP_DIR"

    # Inicializa git e conecta ao remote
    git init
    git remote add origin "$REPO_URL"
    git fetch origin

    # Identifica branch principal do remote
    MAIN_BRANCH=$(git remote show origin 2>/dev/null | grep 'HEAD branch' | awk '{print $NF}')
    MAIN_BRANCH="${MAIN_BRANCH:-main}"

    # Checkout do remote, preservando arquivos locais não-trackeados
    # Usa --no-overlay para não deletar arquivos que não estão no repo
    git checkout -b "$MAIN_BRANCH" "origin/$MAIN_BRANCH" 2>/dev/null || {
        # Se falhar (arquivos conflitantes), faz merge manual
        git checkout "origin/$MAIN_BRANCH" -- .gitignore 2>/dev/null || true
        git reset "origin/$MAIN_BRANCH"
        git checkout -- .gitignore 2>/dev/null || true
    }

    echo "[init] ✓ Repo sincronizado"
    echo "[init] Arquivos do repo foram aplicados. Arquivos locais preservados."
    echo "[init] Backup disponível em: $BACKUP_DIR"
    echo ""
    echo "[init] Verifique com: cd ~/.claude && git status"
    echo "[init] Se tudo OK, pode remover o backup: rm -rf $BACKUP_DIR"
}

if [ "$INIT" = true ] || { [ ! -d "$CLAUDE_DIR/.git" ] && [ -n "$REPO_URL" ]; }; then
    init_repo
fi

# --- Pré-requisitos ---

if ! command -v claude &>/dev/null; then
    error "Claude Code CLI não encontrado. Instale primeiro: https://code.claude.com"
fi

if [ ! -f "$MANIFEST" ]; then
    error "Manifest não encontrado em $MANIFEST. Rode com --init primeiro para sincronizar o repo."
fi

if ! command -v jq &>/dev/null; then
    error "jq não encontrado. Instale: brew install jq (macOS) ou apt install jq (Linux)"
fi

# --- Detectar Node ---

find_best_node() {
    local min_major=18
    local best_node=""
    local best_version=0

    # 1. Tenta nvm
    if [ -d "$HOME/.nvm/versions/node" ]; then
        for node_dir in "$HOME/.nvm/versions/node"/v*/bin/node; do
            if [ -x "$node_dir" ]; then
                local ver
                ver=$("$node_dir" --version 2>/dev/null | sed 's/v//' | cut -d. -f1)
                if [ "$ver" -ge "$min_major" ] 2>/dev/null && [ "$ver" -gt "$best_version" ]; then
                    best_version=$ver
                    best_node=$(dirname "$node_dir")
                fi
            fi
        done
    fi

    # 2. Tenta fnm
    if [ -z "$best_node" ] && [ -d "$HOME/.fnm" ]; then
        for node_dir in "$HOME/.fnm/node-versions"/v*/installation/bin; do
            if [ -x "$node_dir/node" ]; then
                local ver
                ver=$("$node_dir/node" --version 2>/dev/null | sed 's/v//' | cut -d. -f1)
                if [ "$ver" -ge "$min_major" ] 2>/dev/null && [ "$ver" -gt "$best_version" ]; then
                    best_version=$ver
                    best_node="$node_dir"
                fi
            fi
        done
    fi

    # 3. Tenta sistema
    if [ -z "$best_node" ]; then
        local sys_node
        sys_node=$(which node 2>/dev/null || true)
        if [ -n "$sys_node" ]; then
            local ver
            ver=$("$sys_node" --version 2>/dev/null | sed 's/v//' | cut -d. -f1)
            if [ "$ver" -ge "$min_major" ] 2>/dev/null; then
                best_node=$(dirname "$sys_node")
            fi
        fi
    fi

    echo "$best_node"
}

NODE_BIN=$(find_best_node)

if [ -z "$NODE_BIN" ]; then
    error "Node.js >= 18 não encontrado. Instale via nvm: nvm install 22"
fi

NODE_VERSION=$("$NODE_BIN/node" --version)
log "Node detectado: $NODE_VERSION em $NODE_BIN"

# --- Clean ---

if [ "$CLEAN" = true ]; then
    log "=== LIMPANDO MCPs instalados pelo bootstrap ==="

    for mcp_name in $(jq -r '.mcps | keys[]' "$MANIFEST"); do
        if claude mcp get "$mcp_name" &>/dev/null; then
            local_scope=$(jq -r ".mcps.\"$mcp_name\".scope" "$MANIFEST")
            claude mcp remove "$mcp_name" -s "$local_scope" 2>/dev/null && \
                log "Removido: $mcp_name" || \
                log "Não encontrado: $mcp_name"
        fi
    done

    # Remove wrappers
    rm -f "$HOOKS_DIR"/run-*.sh
    rm -f "$LOCAL_STATE"
    log "=== LIMPEZA CONCLUÍDA ==="
    exit 0
fi

# --- Check ---

if [ "$CHECK_ONLY" = true ]; then
    log "=== VERIFICAÇÃO DO AMBIENTE ==="
    log "Node: $NODE_VERSION ($NODE_BIN)"
    log ""
    log "MCPs requeridos:"

    all_ok=true
    for mcp_name in $(jq -r '.mcps | keys[]' "$MANIFEST"); do
        required=$(jq -r ".mcps.\"$mcp_name\".required" "$MANIFEST")
        if claude mcp list 2>/dev/null | grep -q "$mcp_name.*Connected"; then
            log "  ✓ $mcp_name - Conectado"
        else
            log "  ✗ $mcp_name - NÃO conectado (required=$required)"
            all_ok=false
        fi
    done

    log ""
    log "MCPs opcionais:"
    for mcp_name in $(jq -r '.optional_mcps | keys[]' "$MANIFEST"); do
        if claude mcp list 2>/dev/null | grep -q "$mcp_name.*Connected"; then
            log "  ✓ $mcp_name - Conectado"
        else
            log "  - $mcp_name - Não instalado"
        fi
    done

    if [ "$all_ok" = true ]; then
        log ""
        log "✓ Ambiente OK"
        exit 0
    else
        log ""
        log "✗ Rode 'bash ~/.claude/setup/bootstrap.sh' para instalar MCPs faltantes"
        exit 1
    fi
fi

# --- Instalar MCPs ---

log "=== BOOTSTRAP DO AMBIENTE .claude ==="
log "Máquina: $(hostname)"
log "OS: $(uname -s) $(uname -m)"
log "Node: $NODE_VERSION ($NODE_BIN)"
log ""

mkdir -p "$HOOKS_DIR"

for mcp_name in $(jq -r '.mcps | keys[]' "$MANIFEST"); do
    log "--- Instalando MCP: $mcp_name ---"

    scope=$(jq -r ".mcps.\"$mcp_name\".scope" "$MANIFEST")
    command=$(jq -r ".mcps.\"$mcp_name\".command" "$MANIFEST")
    args=$(jq -r ".mcps.\"$mcp_name\".args | join(\" \")" "$MANIFEST")

    # Remove instalação anterior se existir
    claude mcp remove "$mcp_name" -s "$scope" 2>/dev/null || true

    # Testa se npx do sistema funciona
    npx_works=false
    if "$NODE_BIN/npx" --version &>/dev/null; then
        npx_works=true
    fi

    if [ "$npx_works" = true ]; then
        # Tenta instalação direta
        sys_npx="$NODE_BIN/npx"

        # Testa se o MCP inicia
        if timeout 15 bash -c "export PATH=\"$NODE_BIN:\$PATH\"; npx --yes $args &>/dev/null &
            PID=\$!; sleep 8; kill \$PID 2>/dev/null; wait \$PID 2>/dev/null" 2>/dev/null; then

            # Cria wrapper para garantir PATH correto
            wrapper="$HOOKS_DIR/run-${mcp_name}.sh"
            cat > "$wrapper" <<WRAPPER
#!/bin/bash
export PATH="$NODE_BIN:\$PATH"
exec npx --yes $args "\$@"
WRAPPER
            chmod +x "$wrapper"

            claude mcp add --scope "$scope" "$mcp_name" -- "$wrapper" 2>&1
            log "✓ $mcp_name instalado (wrapper: $wrapper)"
        else
            log "✗ $mcp_name: falha no teste de inicialização"
        fi
    else
        log "✗ $mcp_name: npx não funciona em $NODE_BIN"
    fi
done

# --- Salvar estado local ---

cat > "$LOCAL_STATE" <<STATE
{
  "bootstrapped_at": "$(date -u '+%Y-%m-%dT%H:%M:%SZ')",
  "hostname": "$(hostname)",
  "os": "$(uname -s) $(uname -m)",
  "node_bin": "$NODE_BIN",
  "node_version": "$NODE_VERSION"
}
STATE

log ""
log "=== BOOTSTRAP CONCLUÍDO ==="
log ""
log "Verifique com: bash ~/.claude/setup/bootstrap.sh --check"
log "Ou: claude mcp list"
