# Git Workflows - Code Review

Comandos e workflows Git úteis para code review. Todos os comandos assumem que você está no diretório raiz do repositório.

---

## Comandos Básicos de Comparação

### Ver branches disponíveis
```bash
# Branch atual
git branch --show-current

# Todas as branches locais
git branch

# Branches remotas
git branch -r

# Todas as branches (local + remote)
git branch -a
```

---

### Validar se branches existem
```bash
# Verificar se branch existe
git rev-parse --verify {branch_name}

# Exemplo
git rev-parse --verify main
git rev-parse --verify origin/feature/new-api
```

**Exit code:**
- `0` = branch existe
- `128` = branch não existe

---

### Ver commits entre branches
```bash
# Lista commits que estão em compare mas não em base
git log {base}..{compare} --oneline

# Com mais detalhes
git log {base}..{compare} --oneline --graph --decorate

# Apenas mensagens de commit
git log {base}..{compare} --pretty=format:"%s"

# Com autor e data
git log {base}..{compare} --pretty=format:"%h - %an, %ar : %s"
```

---

## Análise de Mudanças

### Estatísticas gerais
```bash
# Resumo de mudanças
git diff --stat {base}..{compare}

# Output exemplo:
# src/api/users.py    | 45 ++++++++++++++++++++++-----
# src/models/user.py  | 12 ++++++--
# tests/test_users.py | 23 +++++++++++++++
# 3 files changed, 71 insertions(+), 9 deletions(-)
```

---

### Lista de arquivos modificados
```bash
# Apenas nomes dos arquivos
git diff --name-only {base}..{compare}

# Com status (M=Modified, A=Added, D=Deleted, R=Renamed)
git diff --name-status {base}..{compare}

# Output exemplo:
# M	src/api/users.py
# A	src/services/auth.py
# D	src/old_module.py
# R100	src/utils.py	src/helpers/utils.py

# Apenas arquivos Python
git diff --name-only {base}..{compare} | grep '\.py$'

# Apenas arquivos de teste
git diff --name-only {base}..{compare} | grep 'test_.*\.py$'
```

---

### Diff completo
```bash
# Diff de todas as mudanças
git diff {base}..{compare}

# Diff de arquivo específico
git diff {base}..{compare} -- {caminho/arquivo.py}

# Diff sem whitespace
git diff -w {base}..{compare}

# Diff com contexto extra (10 linhas antes e depois)
git diff -U10 {base}..{compare}

# Diff mostrando apenas nomes de funções alteradas
git diff {base}..{compare} --function-context
```

---

### Análise por tipo de arquivo
```bash
# Contar arquivos por extensão
git diff --name-only {base}..{compare} | sed 's/.*\.//' | sort | uniq -c | sort -nr

# Output exemplo:
#   12 py
#    3 md
#    2 txt
#    1 toml

# Listar apenas arquivos Python modificados com estatísticas
git diff --stat {base}..{compare} -- '*.py'

# Listar apenas arquivos de teste
git diff --stat {base}..{compare} -- 'tests/test_*.py' '**/test_*.py'
```

---

## Análise Detalhada de Arquivo

### Ver mudanças em arquivo específico
```bash
# Diff do arquivo
git diff {base}..{compare} -- {arquivo}

# Com números de linha
git diff -U3 {base}..{compare} -- {arquivo} | cat -n

# Ver apenas linhas adicionadas
git diff {base}..{compare} -- {arquivo} | grep '^+'

# Ver apenas linhas removidas
git diff {base}..{compare} -- {arquivo} | grep '^-'
```

---

### Blame e histórico
```bash
# Ver quem modificou cada linha (na branch compare)
git blame {compare} -- {arquivo}

# Ver histórico de commits que tocaram o arquivo
git log {base}..{compare} -- {arquivo}

# Ver diff de cada commit que tocou o arquivo
git log -p {base}..{compare} -- {arquivo}
```

---

## Análise de Autores e Atividade

### Autores das mudanças
```bash
# Lista autores únicos
git log {base}..{compare} --format='%an' | sort | uniq

# Contagem por autor
git log {base}..{compare} --format='%an' | sort | uniq -c | sort -rn

# Output exemplo:
#   15 Alice Developer
#    8 Bob Engineer
#    3 Charlie Contributor

# Commits por autor com mensagens
git log {base}..{compare} --format='%an: %s' | sort
```

---

### Data e tempo das mudanças
```bash
# Primeiro e último commit
git log {base}..{compare} --format='%ai %s' | head -1
git log {base}..{compare} --format='%ai %s' | tail -1

# Todos os commits com data
git log {base}..{compare} --format='%ai - %an: %s'
```

---

## Verificações Úteis

### Detectar arquivos grandes adicionados
```bash
# Listar arquivos adicionados maiores que 1MB
git diff {base}..{compare} --stat | awk '{if ($3 ~ /\+/ && $1 ~ /Bin/) print $0}'

# Ver tamanho de arquivos modificados
git diff {base}..{compare} --stat=200
```

---

### Detectar arquivos movidos ou renomeados
```bash
# Ver movimentos/renomeações
git diff {base}..{compare} --name-status | grep '^R'

# Com percentual de similaridade
git diff {base}..{compare} --name-status -M

# Output exemplo:
# R095	src/old_name.py	src/new_name.py
```

---

### Verificar se há merge conflicts
```bash
# Simular merge para detectar conflicts
git merge-tree $(git merge-base {base} {compare}) {base} {compare}

# Mais simples (mas modifica working directory temporariamente)
git checkout {base}
git merge --no-commit --no-ff {compare}
git merge --abort  # desfaz
```

---

## Análise de Conteúdo

### Buscar padrões no diff
```bash
# Buscar por palavra-chave nas mudanças
git diff {base}..{compare} | grep -i "password"
git diff {base}..{compare} | grep -i "api_key"
git diff {base}..{compare} | grep -i "secret"

# Buscar por imports adicionados
git diff {base}..{compare} | grep '^+import'
git diff {base}..{compare} | grep '^+from .* import'

# Buscar por TODOs adicionados
git diff {base}..{compare} | grep '^+.*TODO'

# Buscar por print statements (code smell)
git diff {base}..{compare} | grep '^+.*print('
```

---

### Análise de complexidade de mudanças
```bash
# Linhas adicionadas vs removidas
git diff {base}..{compare} --numstat

# Output exemplo:
# 45	9	src/api/users.py
# 12	3	src/models/user.py
# (45 linhas adicionadas, 9 removidas)

# Total de linhas mudadas
git diff {base}..{compare} --shortstat

# Output exemplo:
# 3 files changed, 71 insertions(+), 9 deletions(-)

# Por arquivo com percentual de mudança
git diff {base}..{compare} --stat=200 --stat-graph-width=20
```

---

## Workflows Avançados

### Comparar com versão específica
```bash
# Comparar branch com commit específico
git diff {commit_hash}..{compare}

# Comparar com tag
git diff v1.0.0..{compare}

# Comparar últimos N commits
git diff HEAD~5..HEAD
```

---

### Ignorar mudanças específicas
```bash
# Ignorar mudanças em whitespace
git diff -w {base}..{compare}

# Ignorar mudanças em linhas em branco
git diff --ignore-blank-lines {base}..{compare}

# Ignorar mudanças em arquivos específicos
git diff {base}..{compare} -- . ':(exclude)package-lock.json' ':(exclude)*.min.js'
```

---

### Exportar diff para análise
```bash
# Salvar diff completo em arquivo
git diff {base}..{compare} > /tmp/review-diff.txt

# Salvar apenas nomes de arquivos
git diff --name-only {base}..{compare} > /tmp/changed-files.txt

# Salvar estatísticas
git diff --stat {base}..{compare} > /tmp/diff-stats.txt

# Salvar diff de cada arquivo Python separadamente
for file in $(git diff --name-only {base}..{compare} | grep '\.py$'); do
    git diff {base}..{compare} -- "$file" > "/tmp/diff-$(basename $file).txt"
done
```

---

## Atalhos e Aliases Úteis

Adicione ao `~/.gitconfig`:
```ini
[alias]
    # Review helpers
    review-files = "!f() { git diff --name-status $1..$2; }; f"
    review-stat = "!f() { git diff --stat $1..$2; }; f"
    review-py = "!f() { git diff --name-only $1..$2 | grep '\\.py$'; }; f"
    review-authors = "!f() { git log $1..$2 --format='%an' | sort | uniq -c | sort -rn; }; f"
    
    # Detect code smells
    review-todos = "!f() { git diff $1..$2 | grep '^+.*TODO'; }; f"
    review-prints = "!f() { git diff $1..$2 | grep '^+.*print('; }; f"
    review-secrets = "!f() { git diff $1..$2 | grep -iE 'password|secret|api_key|token'; }; f"
```

**Uso:**
```bash
git review-files origin/main feature/new-api
git review-py origin/main HEAD
git review-secrets origin/main feature/new-api
```

---

## Padrões de Uso no Review-Py

### Workflow típico
```bash
# 1. Validar branches
git rev-parse --verify {base}
git rev-parse --verify {compare}

# 2. Obter estatísticas gerais
git diff --stat {base}..{compare}

# 3. Listar arquivos Python modificados
git diff --name-only {base}..{compare} | grep '\.py$'

# 4. Para cada arquivo Python:
#    a. Ver diff
git diff {base}..{compare} -- {arquivo}

#    b. Executar análise (script Python)
python scripts/analyze_diff.py --file {arquivo} --base {base} --compare {compare}

# 5. Gerar report
python scripts/format_output.py --output review-output.md
```

---

### Quick checks antes do review
```bash
# Verificar se há muitas mudanças (>1000 linhas)
CHANGES=$(git diff {base}..{compare} --shortstat | grep -oE '[0-9]+ insertion' | grep -oE '[0-9]+')
if [ "$CHANGES" -gt 1000 ]; then
    echo "⚠️ Atenção: PR muito grande ($CHANGES linhas). Considere quebrar."
fi

# Verificar se há arquivos não-Python
NON_PY=$(git diff --name-only {base}..{compare} | grep -v '\.py$' | wc -l)
if [ "$NON_PY" -gt 0 ]; then
    echo "ℹ️ $NON_PY arquivos não-Python modificados"
fi

# Verificar se há novos arquivos
NEW_FILES=$(git diff --name-status {base}..{compare} | grep '^A' | wc -l)
if [ "$NEW_FILES" -gt 0 ]; then
    echo "✨ $NEW_FILES novos arquivos adicionados"
fi
```

---

## Troubleshooting

### Branch não encontrada
```bash
# Fetch branches remotas mais recentes
git fetch origin

# Verificar se branch existe remotamente
git ls-remote --heads origin {branch_name}

# Criar tracking branch local se necessário
git checkout -b {local_name} origin/{remote_name}
```

---

### Diff muito grande
```bash
# Ver apenas estatísticas sem diff completo
git diff --stat {base}..{compare}

# Ver diff de arquivos menores primeiro
git diff {base}..{compare} --stat | awk '$3 ~ /\+/ {print $3, $1}' | sort -n

# Limitar diff a N linhas de contexto
git diff -U1 {base}..{compare}
```

---

### Performance em repos grandes
```bash
# Usar shallow diff (apenas arquivos mudados)
git diff --name-only {base}..{compare}

# Desabilitar renames detection (mais rápido)
git diff --no-renames {base}..{compare}

# Limitar profundidade do log
git log {base}..{compare} --oneline -n 100
```

---

## Comandos NÃO Recomendados

**Evite modificar o repositório durante review:**
```bash
# ❌ NÃO FAZER - checkout modifica working directory
git checkout {compare}

# ❌ NÃO FAZER - merge modifica histórico
git merge {compare}

# ❌ NÃO FAZER - rebase reescreve histórico
git rebase {base}

# ❌ NÃO FAZER - reset perde mudanças
git reset --hard
```

**Review deve ser read-only!**

---

## Referências

- Git Diff Documentation: https://git-scm.com/docs/git-diff
- Git Log Documentation: https://git-scm.com/docs/git-log
- Pro Git Book: https://git-scm.com/book/en/v2
- Git Best Practices: https://sethrobertson.github.io/GitBestPractices/

---

## Notas Importantes

**Sobre branches:**
- Use `origin/{branch}` para branches remotas
- Use `{branch}` para branches locais
- `HEAD` sempre referencia o commit atual
- `HEAD~N` referencia N commits atrás

**Sobre performance:**
- Diffs grandes (>1000 arquivos) podem ser lentos
- Use `--stat` para overview rápido primeiro
- Considere revisar em batches menores

**Sobre segurança:**
- Git commands no review são read-only
- Nunca execute comandos que modificam o repo
- Sempre valide branches antes de comparar
```

---