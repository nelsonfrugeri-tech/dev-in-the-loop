---
name: review-py
description: |
  Code review sistemático para projetos Python entre branches Git.
  Analisa impacto de mudanças, gera review detalhado arquivo por arquivo, e produz comentários formatados para Bitbucket/GitHub/GitLab.
  Integra com developer skill para referenciar best practices Python.
  Use quando: (1) Revisar PR/MR, (2) Analisar mudanças entre branches, (3) Code review antes de merge.
  Triggers: /review, /review-py, code review, PR review, merge request, revisar código.
---

# Review-Py Skill - Python Code Review System

## Padrão de Conversa

### Princípios de Comunicação

**Verificabilidade e Transparência:**
- Baseie toda análise em código real extraído via `git diff`
- Nunca invente problemas que não existem no diff real
- Se não puder verificar algo diretamente no código, diga claramente
- Rotule inferências com `[Inference]` quando aplicável

**Objetividade e Acionabilidade:**
- Comentários devem ser específicos e acionáveis
- Sempre mostre "Código Atual" vs "Código Sugerido"
- Explique o "porquê" da sugestão, não apenas o "o quê"
- Referencie linhas e arquivos específicos

**Integração com Developer Skill:**
- Referencie developer skill quando encontrar violação de best practice
- Use developer skill como "source of truth" para padrões Python
- Cite arquivos específicos: `[references/python/type-system.md](../developer/references/python/type-system.md)`

**Preservação de Input:**
- Nunca altere branches informadas pelo usuário
- Use exatamente as branches fornecidas nos comandos git

---

## Workflow Principal

Quando invocado com `/review` ou `/review-py`:

### Step 0: Configuração de Branches

**Detectar branch atual:**
````bash
git branch --show-current
git branch -r | head -10
````

**Apresentar ao usuário:**
````
🔍 Review-Py Setup

Branch atual detectada: feature/new-endpoint
Branches remotas disponíveis:
  - origin/main
  - origin/develop
  - origin/staging

Digite as branches para comparação:
- Base branch (ex: main, origin/main): _______
- Compare branch (ex: feature/xyz, HEAD): _______

[Enter para usar: compare=HEAD, base=main]
````

**Validar branches:**
````bash
git rev-parse --verify {base}
git rev-parse --verify {compare}
````

Se inválidas, informar erro e pedir novamente.

---

### Step 1: Menu Interativo

Após branches confirmadas:
````
┌──────────────────────────────────────────────────────────┐
│ 🔍 Review-Py - Python Code Review System                 │
├──────────────────────────────────────────────────────────┤
│ Comparando: {compare} → {base}                           │
│                                                           │
│ Escolha uma opção:                                        │
│                                                           │
│ [1] 📊 Análise de Impacto                                │
│     • Estatísticas das mudanças                          │
│     • Features identificadas                             │
│     • Divisão por áreas do código                        │
│     • Recomendações de prioridade                        │
│                                                           │
│ [2] 📝 Review por Arquivo                                │
│     • Lista arquivos Python modificados                  │
│     • Review detalhado linha por linha                   │
│     • Comentários formatados (Bitbucket-ready)           │
│                                                           │
│ [3] 📋 Relatório Completo                                │
│     • Análise de impacto + Review todos arquivos         │
│     • Salva tudo em review-output.md                     │
│                                                           │
│ [4] ⚙️  Trocar Branches                                  │
│                                                           │
└──────────────────────────────────────────────────────────┘

Digite o número da opção: _____
````

---

## Opção 1: Análise de Impacto

### Comandos Git

Execute sequencialmente:
````bash
# 1. Estatísticas gerais
git diff --stat {base}..{compare}

# 2. Lista de arquivos com status
git diff --name-status {base}..{compare}

# 3. Diff completo
git diff {base}..{compare}
````

### Análise com Script

Execute o script de análise:
````bash
python scripts/analyze_diff.py --base {base} --compare {compare} --output json
````

O script retorna:
- Total de arquivos por tipo (.py, .txt, .md, etc)
- Métricas de complexidade
- Padrões detectados (imports, docstrings, type hints coverage)
- Features identificadas (agrupamento lógico de arquivos)

### Output Gerado

Use o template `assets/summary.md` e preencha com os dados:
````markdown
## 📊 Análise de Impacto das Mudanças

**Branches:** `{compare}` → `{base}`  
**Data:** {timestamp}  
**Reviewer:** Claude Code (review-py skill)

---

### Estatísticas Gerais
- **Total de arquivos:** {total}
- **Arquivos Python:** {python_count} (.py)
- **Arquivos de teste:** {test_count} (test_*.py)
- **Config/Deps:** {config_count} (requirements.txt, pyproject.toml, etc)
- **Linhas adicionadas:** +{additions}
- **Linhas removidas:** -{deletions}
- **Impacto estimado:** {impacto} (Baixo/Médio/Alto)

---

### Arquivos Modificados por Categoria

#### 🔧 Core Application ({count} arquivos)
{lista de arquivos com (+X, -Y) e status}

#### ✅ Tests ({count} arquivos)
{lista de arquivos de teste}

#### 📦 Dependencies ({count} arquivos)
{requirements.txt, pyproject.toml, etc}

#### 📝 Documentation ({count} arquivos)
{README, docs, etc}

---

### Features Identificadas

Para cada feature detectada pelo script:

**Feature #{n}: {nome_da_feature}**
- **Arquivos:** {lista}
- **Impacto:** {baixo/médio/alto} ({razão})
- **Risco:** {baixo/médio/alto} ({razão})
- **Mudanças:**
  - {mudança 1}
  - {mudança 2}

---

### Recomendações de Review

**Prioridade Alta (revisar primeiro):**
{arquivos críticos - novos, segurança, core logic}

**Prioridade Média:**
{arquivos importantes - models, schemas, validações}

**Prioridade Baixa:**
{testes, docs, configs simples}

---

### Próximos Passos
→ Selecione opção [2] para review detalhado por arquivo  
→ Ou [3] para relatório completo com todos os comentários
````

**Critérios de Impacto:**
- **Alto:** Novos arquivos críticos, mudanças em auth/segurança, schema changes
- **Médio:** Modificações em core logic, novas features
- **Baixo:** Testes, docs, refactoring sem mudança de comportamento

---

## Opção 2: Review por Arquivo

### Step 2.1: Listar Arquivos Python
````bash
git diff --name-only {base}..{compare} | grep '\.py$'
````

### Step 2.2: Apresentar Lista
````
📝 Arquivos Python Modificados:

[1] src/api/endpoints/users.py       (+87, -12)  M
[2] src/models/user.py                (+34, -8)   M
[3] src/schemas/user.py               (+45, -15)  M
[4] src/services/auth.py              (+56, -0)   A (novo)
[5] tests/test_users.py               (+78, -20)  M
[6] tests/test_auth.py                (+89, -0)   A (novo)

Digite:
- Número do arquivo (ex: 1)
- Múltiplos números separados por vírgula (ex: 1,4,5)
- "all" para todos
- "critical" para apenas novos e modificações em core (sugerido: 1,2,4)

Sua escolha: _____
````

### Step 2.3: Review de Cada Arquivo

Para cada arquivo selecionado:

**1. Obter diff do arquivo:**
````bash
git diff {base}..{compare} -- {arquivo}
````

**2. Executar análise automática:**
````bash
python scripts/analyze_diff.py --file {arquivo} --base {base} --compare {compare}
````

O script detecta automaticamente:
- Type hints faltando
- Docstrings ausentes
- Secrets hardcoded (regex patterns)
- N+1 query patterns (loops com queries)
- Exception handling inadequado
- Imports não utilizados
- Complexity metrics (cyclomatic complexity)

**3. Consultar checklist manual:**

Leia `references/checklist.md` e verifique cada item aplicável.

**4. Gerar comentários:**

Para cada issue encontrado, use o template `assets/comment.md`:
````markdown
### Comentário #{n}

**Linhas:** {start_line}-{end_line}  
**Categoria:** {categoria_emoji} {categoria_nome}  
**Severidade:** {severidade_emoji} {severidade_nome}

**Issue:**
{descrição clara e objetiva do problema}

**Código Atual:**
```python
{código problemático extraído do diff}
```

**Código Sugerido:**
```python
{código corrigido}
```

**Justificativa:**
{explicação técnica do porquê isso é um problema}
{impacto se não corrigir}

**Referência:**
- Developer Skill: [{arquivo_referencia}](../developer/{arquivo_referencia})
{referências externas se aplicável}

---
````

**Categorias disponíveis:**
- 🔒 Security
- ⚡ Performance
- 🧪 Testing
- 📝 Documentation
- ⚡ Code Quality
- 🏗️ Architecture

**Severidades disponíveis:**
Consulte `references/severity-levels.md` para critérios exatos.
- 🔴 Critical
- 🟠 High
- 🟡 Medium
- 🟢 Low
- ℹ️ Info

**5. Pontos Positivos:**

Sempre inclua seção de pontos positivos:
````markdown
### ✅ Pontos Positivos

1. ✨ {aspecto bem feito}
2. ✨ {boas práticas seguidas}
````

**6. Resumo do Arquivo:**
````markdown
### 📊 Resumo do Arquivo

| Categoria | Count | Severidade Máxima |
|-----------|-------|-------------------|
| 🔒 Security | {n} | {max} |
| ⚡ Performance | {n} | {max} |
| 🧪 Testing | {n} | {max} |
| ⚡ Code Quality | {n} | {max} |
| **Total** | **{total}** | - |

**Recomendação:** {aprovar/não aprovar/aprovar com ressalvas}
**Justificativa:** {razão da recomendação}
````

### Step 2.4: Salvar Output

Após revisar todos os arquivos selecionados:
````bash
python scripts/format_output.py \
  --reviews {arquivos_json_gerados} \
  --output review-output.md \
  --format bitbucket
````

Informar ao usuário:
````
✅ Review completo salvo em: review-output.md
📋 {total} comentários gerados em {n} arquivos
🔴 {critical} issues críticos encontrados

O arquivo está pronto para copy-paste no Bitbucket.
````

---

## Opção 3: Relatório Completo

Executa automaticamente:

1. **Opção 1** (Análise de Impacto) → salva em memória
2. **Opção 2** (Review de TODOS arquivos .py) → salva em memória
3. Combina tudo em `review-output.md` usando `assets/report.md`

### Estrutura do Relatório
````markdown
# Code Review Report

**Date:** {timestamp}  
**Branches:** {compare} → {base}  
**Reviewer:** Claude Code (review-py skill)

---

{conteúdo completo da Análise de Impacto}

---

{review de cada arquivo Python}

---

## 📊 Resumo Geral

### Por Severidade
- 🔴 Critical: {n} issues
- 🟠 High: {n} issues
- 🟡 Medium: {n} issues
- 🟢 Low: {n} issues
- ℹ️ Info: {n} issues

### Por Categoria
- 🔒 Security: {n} issues
- ⚡ Performance: {n} issues
- 🧪 Testing: {n} issues
- 📝 Documentation: {n} issues
- ⚡ Code Quality: {n} issues

### Arquivos Revisados
- Total: {n}
- Com issues: {n}
- Sem issues: {n}

### Recomendação Final
{emoji} **{decisão}**

{se não aprovar, listar issues bloqueantes}

### Issues Bloqueantes (se aplicável)
1. `{arquivo}:{linha}` - {descrição curta} ({severidade})
2. `{arquivo}:{linha}` - {descrição curta} ({severidade})

---

**Relatório gerado por:** review-py skill  
**Formato:** Copy-paste ready para Bitbucket/GitHub/GitLab
````

---

## Integração com Developer Skill

### Referências Automáticas

Quando detectar violação de best practice da developer skill, sempre referencie:

**Exemplos:**

**Type hints faltando:**
````markdown
**Referência:**
- Developer Skill: [references/python/type-system.md](../developer/references/python/type-system.md)
````

**Error handling inadequado:**
````markdown
**Referência:**
- Developer Skill: [references/python/error-handling.md](../developer/references/python/error-handling.md)
````

**Logging sem estrutura:**
````markdown
**Referência:**
- Developer Skill: [references/python/logging.md](../developer/references/python/logging.md)
````

### Checklist Baseado em Developer Skill

O arquivo `references/checklist.md` está mapeado com a developer skill.

Para cada categoria da developer skill, há checks correspondentes no review.

---

## Comandos Git Úteis

### Referência Rápida
````bash
# Ver arquivos modificados
git diff --name-only {base}..{compare}

# Ver arquivos Python modificados
git diff --name-only {base}..{compare} | grep '\.py$'

# Ver diff de arquivo específico
git diff {base}..{compare} -- {arquivo}

# Ver estatísticas
git diff --stat {base}..{compare}

# Ver commits entre branches
git log {base}..{compare} --oneline

# Ver autores das mudanças
git log {base}..{compare} --format='%an' | sort | uniq -c

# Ver apenas mudanças em código (sem whitespace)
git diff -w {base}..{compare}

# Ver mudanças com contexto extra (10 linhas)
git diff -U10 {base}..{compare}
````

Consulte `references/git-workflows.md` para workflows avançados.

---

## Estrutura de Arquivos da Skill
````
review-py/
├── SKILL.md                          (este arquivo)
├── references/
│   ├── checklist.md                 (checklist completo mapeado com developer)
│   ├── severity-levels.md           (critérios de classificação detalhados)
│   ├── comment-templates.md         (exemplos de comentários bem feitos)
│   └── git-workflows.md             (comandos git avançados)
├── scripts/
│   ├── analyze_diff.py              (parsing e análise de diffs)
│   └── format_output.py             (formatação de comentários)
└── assets/
    ├── comment.md                   (template de comentário individual)
    ├── summary.md                   (template de análise de impacto)
    └── report.md                    (template de relatório completo)
````

---

## Referências

### Arquivos desta Skill
- [references/checklist.md](references/checklist.md) - Checklist completo
- [references/severity-levels.md](references/severity-levels.md) - Critérios de severidade
- [references/comment-templates.md](references/comment-templates.md) - Exemplos de comentários
- [references/git-workflows.md](references/git-workflows.md) - Workflows Git avançados

### Developer Skill (Best Practices)
- [../developer/SKILL.md](../developer/SKILL.md) - Developer skill principal
- [../developer/references/python/](../developer/references/python/) - Padrões Python
- [../developer/references/testing/](../developer/references/testing/) - Padrões de testes
- [../developer/references/architecture/](../developer/references/architecture/) - Arquitetura

### Scripts
- [scripts/analyze_diff.py](scripts/analyze_diff.py) - Análise automática de diffs
- [scripts/format_output.py](scripts/format_output.py) - Formatação de output

### Templates
- [assets/comment.md](assets/comment.md) - Template de comentário
- [assets/summary.md](assets/summary.md) - Template de summary
- [assets/report.md](assets/report.md) - Template de relatório