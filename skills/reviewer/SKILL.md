---
name: review-py
description: |
  Code review sistemático para projetos Python entre branches Git.
  Gera análise de impacto, review detalhado arquivo por arquivo, e comentários formatados para copy-paste em PRs.
  Integra com arch-py skill para referenciar best practices Python.
  Use quando: (1) Revisar PR/MR Python, (2) Analisar mudanças antes de merge, (3) Code review entre branches.
  Triggers: /review, /review-py, code review, PR review, merge request, revisar código.
---

# Review-Py Skill - Python Code Review System

## Padrão de Conversa

### Princípios de Comunicação

**Verificabilidade e Transparência:**
- Baseie análises em código real extraído via `git diff`
- Nunca invente problemas que não existem no diff
- Se não puder verificar algo diretamente no código, diga claramente
- Rotule inferências com `[Inference]` quando aplicável

**Objetividade:**
- Comentários devem ser acionáveis e específicos
- Sempre mostre código atual vs código sugerido
- Explique o "porquê" da sugestão, não apenas o "o quê"

**Integração:**
- Referencie arch-py skill quando aplicável
- Cite linhas e arquivos específicos
- Mantenha rastreabilidade do feedback

### Uso de Assets e Scripts

**Assets (Templates):**
- São templates markdown que você LEIA com `view` e PREENCHA os placeholders
- `assets/comment.md` → template de comentário individual
- `assets/summary.md` → template de análise de impacto
- `assets/report.md` → template de relatório completo
- **IMPORTANTE:** Você preenche e apresenta o resultado final ao usuário, não apenas cita o template

**Scripts Python:**
- `analyze_diff.py` → análise automática de diffs (métricas, padrões, features)
- `format_output.py` → compilador opcional de JSON → markdown
- Use scripts para acelerar análise, mas review manual é sempre necessário

**References (Documentação):**
- `references/checklist.md` → checklist de review com ponteiros para arch-py skill
- `references/templates.md` → exemplos de comentários por tipo de issue
- `references/git.md` → comandos git úteis
- **Consulte** quando precisar de detalhes, exemplos ou comandos específicos

---

## Workflow Principal

Quando invocado com `/review` ou `/review-py`, inicie este fluxo:

### Step 0: Detectar ou Solicitar Branches

Execute primeiro:
```bash
git branch --show-current
git branch -r | head -10
```

Se branches claras, pergunte:
```
🔍 Branches detectadas:
• Atual: {current_branch}
• Remotas disponíveis: {lista}

Digite as branches para comparação:
Base branch (ex: main, develop): _______
Compare branch (ex: feature/xyz): _______
```

Armazene as branches escolhidas como variáveis: `{base}` e `{compare}`

---

### Step 1: Menu de Opções

Após branches definidas, apresentar:

```
┌──────────────────────────────────────────────────────────┐
│ 🔍 Review-Py - Code Review Python                        │
├──────────────────────────────────────────────────────────┤
│ Comparando: {compare} → {base}                           │
│                                                           │
│ Escolha uma opção:                                        │
│                                                           │
│ [1] 📊 Análise de Impacto                                │
│     • Estatísticas das mudanças                          │
│     • Features identificadas                             │
│     • Divisão por áreas do código                        │
│                                                           │
│ [2] 📝 Review por Arquivo                                │
│     • Lista arquivos modificados                         │
│     • Review detalhado com comentários                   │
│     • Formato copy-paste para PR                         │
│                                                           │
│ [3] 📋 Relatório Completo                                │
│     • Análise de impacto + Review todos arquivos         │
│     • Salva tudo em review-output.md                     │
│                                                           │
│ [4] ⚙️  Trocar Branches                                  │
│                                                           │
└──────────────────────────────────────────────────────────┘

Digite o número da opção: _____
```

---

## Opção 1: Análise de Impacto

### Comandos a Executar

```bash
# 1. Estatísticas gerais
git diff --stat {base}..{compare}

# 2. Lista de arquivos com status
git diff --name-status {base}..{compare}

# 3. Diff completo para análise
git diff {base}..{compare}

# 4. Filtrar apenas Python
git diff --name-only {base}..{compare} | grep '\.py$'
```

### Processo de Análise

1. **Execute os comandos acima**

2. **Execute script de análise:**
```bash
python scripts/analyze_diff.py --base {base} --compare {compare} --format summary
```

O script retorna JSON com:
- `total_files`: número total de arquivos modificados
- `python_files`: lista de arquivos .py
- `stats`: {additions, deletions, net_change}
- `features`: features identificadas automaticamente
- `complexity_metrics`: métricas por arquivo
- `alerts`: alertas automáticos (secrets, patterns)

3. **Leia o template base:**
```bash
view assets/summary.md
```

4. **Preencha os placeholders do template:**

**Placeholders obrigatórios:**
- `{base_branch}` → nome da base branch
- `{compare_branch}` → nome da compare branch
- `{review_date}` → data atual (ex: "2024-02-07 14:30")
- `{total_commits}` → do `git log {base}..{compare} --oneline | wc -l`
- `{total_files}` → do script analyze_diff.py
- `{python_files}` → do script
- `{lines_added}` → do `git diff --shortstat`
- `{lines_removed}` → do `git diff --shortstat`
- `{net_change}` → diferença entre added e removed

**Placeholders de listas:**
- `{python_modified_count}` e `{python_modified_list}` → arquivos .py modificados
- `{python_added_count}` e `{python_added_list}` → arquivos .py novos
- `{python_deleted_count}` e `{python_deleted_list}` → arquivos .py deletados
- `{python_renamed_count}` e `{python_renamed_list}` → arquivos .py renomeados
- `{other_files_count}` e `{other_files_list}` → outros arquivos

**Placeholders de análise:**
- `{features_list}` → do script analyze_diff.py (features detectadas)
- `{authors_list}` → do `git log --format='%an' | sort | uniq -c`
- `{complexity_table}` → tabela markdown com dados do script
- `{preliminary_alerts}` → alertas do script (secrets, patterns)

**Placeholders de priorização:**
- `{high_priority_files}` → arquivos críticos (novos, auth, schemas)
- `{medium_priority_files}` → arquivos importantes (models, apis)
- `{low_priority_files}` → arquivos menos críticos (tests, docs)

**Placeholders de next steps:**
- `{next_step_1}` → geralmente "Execute opção [2] para review detalhado"
- `{next_step_2}` → "Ou [3] para relatório completo"
- `{next_step_3}` → dica adicional se aplicável

5. **Gere o output final:**

Usando o template lido de `assets/summary.md`, substitua TODOS os placeholders pelos dados coletados nos steps anteriores. O output final deve:
- Seguir exatamente a estrutura markdown do template
- Ter todos os placeholders substituídos por valores reais
- Manter as seções e formatação do template
- Ser apresentado ao usuário em formato markdown completo

**IMPORTANTE:** Você está PREENCHENDO o template, não apenas citando-o. O usuário deve ver o summary completo e formatado.

6. **Salvar output (opcional):**

Se o usuário pedir para salvar:
```bash
# Salvar em arquivo
cat > review-output.md << 'EOF'
{todo o conteúdo formatado}
EOF
```

7. **Referências úteis:**
   - Consulte `references/checklist.md` para severidade típica de cada tipo de issue
   - Use critérios da arch-py skill para avaliar complexidade

---

## Opção 2: Review por Arquivo

### Processo Detalhado

#### 1. Listar Arquivos Python Modificados

```bash
git diff --name-only {base}..{compare} | grep '\.py$'
```

Apresente lista numerada:
```
📝 Arquivos Python Modificados:

[1] src/api/endpoints/users.py       (+87, -12)
[2] src/models/user.py                (+34, -8)
[3] src/schemas/user.py               (+45, -15)
[4] src/services/auth.py              (+56, -0) NEW
[5] tests/test_users.py               (+78, -20)
[6] tests/test_auth.py                (+89, -0) NEW

Digite o número do arquivo para revisar (ou "all" para todos): _____
```

#### 2. Para Cada Arquivo Selecionado

**a. Obter diff do arquivo:**
```bash
git diff {base}..{compare} -- {filepath}
```

**b. Executar análise automatizada:**
```bash
python scripts/analyze_diff.py --file {filepath} --base {base} --compare {compare}
```

**c. Consultar checklist de review:**
- Leia `references/checklist.md` (checklist lean com ponteiros para arch-py skill)
- Verifique cada item aplicável ao arquivo
- Para detalhes de padrões, consulte arch-py skill conforme referenciado no checklist

**d. Gerar comentários:**

Para cada issue encontrado:

**d.1) Leia o template base:**
```bash
view assets/comment.md
```

**d.2) Preencha os placeholders:**

**Identificação:**
- `{comment_number}` → número sequencial (1, 2, 3...)
- `{start_line}` → linha inicial do código problemático
- `{end_line}` → linha final do código problemático

**Classificação:**
- `{category_emoji}` → emoji da categoria (🔒, ⚡, 🧪, 📝, ⚙️, 🏗️)
- `{category_name}` → nome da categoria (Security, Performance, Testing, Documentation, Code Quality, Architecture)
- `{severity_emoji}` → emoji severidade (🔴, 🟠, 🟡, 🟢, ℹ️)
- `{severity_name}` → nome severidade (Critical, High, Medium, Low, Info)

**Conteúdo:**
- `{issue_description}` → descrição clara do problema em 1-2 frases
- `{current_code}` → código problemático extraído do diff (sem ```python)
- `{suggested_code}` → código corrigido/melhorado (sem ```python)
- `{justification}` → explicação técnica do porquê em 2-3 parágrafos

**Seções opcionais (use quando aplicável):**
- `{impact_section}` → (para Critical/High) explicar impacto se não corrigir
  - Formato: "**Impacto:** {descrição do impacto em produção}"
- `{action_required}` → (para Critical) adicionar nota de bloqueio
  - Formato: "**Ação Requerida:** Bloqueia merge. Deve ser corrigido imediatamente."
- `{references}` → links para arch-py skill e docs externas
  - Sempre incluir link para arch-py skill quando aplicável
  - Formato: "- Arch-Py Skill: [{arquivo}](../arch-py/references/{caminho})"

**EXEMPLO DE PREENCHIMENTO:**

Template original:
```markdown
### Comentário #{comment_number}
**Linhas:** {start_line}-{end_line}  
**Categoria:** {category_emoji} {category_name}  
**Severidade:** {severity_emoji} {severity_name}
**Issue:** {issue_description}
...
```

Template preenchido:
```markdown
### Comentário #1
**Linhas:** 42-45  
**Categoria:** 🔒 Security  
**Severidade:** 🔴 Critical
**Issue:** Secret key hardcoded no código. Credenciais nunca devem estar no código fonte.
...
```

**d.3) Para exemplos de comentários bem formatados:**
```bash
view references/templates.md
```

Este arquivo contém templates específicos por:
- Severidade (Critical, High, Medium, Low, Info)
- Categoria (Security - SQL Injection, Performance - N+1, etc)
- Tipo de issue comum

**d.4) Classifique severidade corretamente:**

Consulte `references/checklist.md` para severidade típica de cada tipo de issue.

**Critérios gerais:**
- 🔴 **Critical:** Vulnerabilidades, secrets expostos, data loss
- 🟠 **High:** Performance grave, falta testes críticos, bugs sérios
- 🟡 **Medium:** Code quality, falta type hints, naming
- 🟢 **Low:** Sugestões de melhoria, optimizações menores
- ℹ️ **Info:** Contexto adicional, FYI

**e. Adicionar pontos positivos:**

Sempre inclua seção de pontos positivos ao final do review do arquivo:

```markdown
### ✅ Pontos Positivos

1. ✨ {aspecto bem implementado}
2. ✨ {boa prática seguida}
3. ✨ {qualidade destacada}
```

**f. Gerar resumo do arquivo:**

```markdown
### 📊 Resumo: `{filepath}`

| Categoria | Count | Severidade Máxima |
|-----------|-------|-------------------|
| {categoria} | {n} | {max_severity} |
| **Total** | **{total}** | **{overall_max}** |

**Recomendação:** {✅ Aprovar / ⚠️ Aprovar com ressalvas / ❌ Não aprovar}
**Justificativa:** {razão da recomendação}
```

**Critérios de recomendação:**
- ❌ **Não aprovar:** 1+ issues Critical
- ⚠️ **Aprovar com ressalvas:** 1+ issues High (corrigir antes de produção)
- ✅ **Aprovar:** Apenas Medium/Low/Info

#### 3. Montar Output Final do Arquivo

Para cada arquivo revisado, compile o output completo seguindo esta estrutura:

```markdown
## 📝 Review: `{filepath}`

**Linhas modificadas:** +{add} -{del}  
**Complexidade:** {baixa/média/alta}

---

{TODOS os comentários gerados (passo d)}

---

{Pontos positivos (passo e)}

---

{Resumo do arquivo (passo f)}

---
```

**IMPORTANTE:** Monte este output COMPLETO em memória. Você vai precisar dele no próximo passo.

#### 4. Salvar ou Acumular Reviews

**Se revisando múltiplos arquivos:**
- Mantenha todos os reviews em memória
- Ao final de TODOS os arquivos, salve tudo junto

**Se revisando apenas 1 arquivo:**
- Salve imediatamente em `review-output.md`

**Como salvar:**

```bash
# Opção A: Salvar manualmente (simples)
cat > review-output.md << 'EOF'
{todos os reviews montados}
EOF

# Opção B: Usar script format_output.py (se tiver dados em JSON)
python scripts/format_output.py \
  --comments {arquivo.json} \
  --output review-output.md \
  --format bitbucket
```

**Quando usar cada opção:**
- Use **Opção A** (manual) quando gerar reviews diretamente em markdown
- Use **Opção B** (script) quando tiver dados estruturados em JSON do analyze_diff.py

#### 5. Informar ao Usuário
```
✅ Review salvo em: review-output.md
📋 {total} comentários em {n} arquivos
🔴 {critical} Critical | 🟠 {high} High | 🟡 {medium} Medium

Arquivo pronto para copy-paste no PR.
```

#### 4. Categorias e Ícones

Use estas categorias (exemplos em `references/templates.md`):
- 🔒 **Security** - Vulnerabilidades, secrets, injeções
- ⚡ **Performance** - N+1 queries, algoritmos ineficientes
- 🧪 **Testing** - Falta de testes, assertions fracas
- 📝 **Documentation** - Docstrings, comentários
- ⚙️ **Code Quality** - Type hints, naming, complexidade
- 🏗️ **Architecture** - SOLID, patterns, acoplamento

#### 5. Integração com Developer Skill

Sempre que identificar violação de padrão Python, referencie:

**Exemplos:**
- Falta type hints → `[references/python/type-system.md](../arch-py/references/python/type-system.md)`
- Error handling ruim → `[references/python/error-handling.md](../arch-py/references/python/error-handling.md)`
- Async incorreto → `[references/python/async-patterns.md](../arch-py/references/python/async-patterns.md)`
- Pydantic errado → `[references/python/pydantic.md](../arch-py/references/python/pydantic.md)`

---

## Opção 3: Relatório Completo

### Processo

1. **Execute Opção 1** (Análise de Impacto) → salve resultado em memória
2. **Execute Opção 2** para TODOS os arquivos .py → salve todos reviews em memória
3. **Compile usando template de relatório**
4. **Salve em `review-output.md`**

### Comandos de Execução

```bash
# 1. Gerar análise completa
python scripts/analyze_diff.py --base {base} --compare {compare} --format full

# 2. Formatar output final usando template
python scripts/format_output.py \
  --base {base} \
  --compare {compare} \
  --analysis {analysis_json} \
  --reviews {reviews_json} \
  --template assets/report.md \
  --output review-output.md
```

### Compilação Manual do Relatório

Se não usar script format_output.py, siga estes passos:

**a) Leia o template de relatório:**
```bash
view assets/report.md
```

**b) Preencha os placeholders principais:**

**Executive Summary:**
- `{files_reviewed}` → total de arquivos .py revisados
- `{total_comments}` → soma de todos os comentários
- `{critical_count}`, `{high_count}`, `{medium_count}`, `{low_count}`, `{info_count}` → contagens por severidade

**Recomendação Final:**
- `{final_recommendation_emoji}` → ✅, ⚠️, ou ❌
- `{final_recommendation_text}` → "Aprovar", "Aprovar com ressalvas", "Não aprovar"
- `{final_justification}` → explicação da decisão baseada nos issues encontrados

**Análise de Impacto (copiar da Opção 1):**
- `{total_commits}`, `{total_files}`, `{python_files}`, etc.
- `{features_list}` → features identificadas
- `{authors_list}` → lista de autores

**Reviews Detalhados:**
- `{detailed_reviews}` → concatenação de todos os reviews da Opção 2

**Resumo por Categoria:**

Para cada categoria (Security, Performance, Testing, Quality, Architecture, Documentation):
- `{category_count}` → total de issues nesta categoria
- `{category_critical}`, `{category_high}`, etc. → contagem por severidade
- `{category_critical_files}`, etc. → lista de arquivos afetados
- `{category_top_issues}` → top 3-5 issues mais importantes

**Action Items por Prioridade:**
- `{blocking_items}` → lista de Critical issues (formato: `arquivo:linha - descrição`)
- `{high_priority_items}` → lista de High issues
- `{medium_priority_items}` → lista de Medium issues
- `{low_priority_items}` → lista de Low e Info

**Destaques Positivos:**
- `{positive_highlights}` → agregação dos pontos positivos de todos arquivos

**Métricas de Qualidade:**
- `{issues_per_file}` → média de issues por arquivo
- `{critical_high_percentage}` → % de issues Critical+High sobre total
- `{estimated_coverage}` → estimativa de cobertura de testes
- `{avg_complexity}` → complexidade média dos arquivos
- `{type_hints_coverage}` → % de type hints presentes

Para cada métrica, adicione status:
- 🟢 Excelente, 🟡 Atenção, 🔴 Crítico

**Análise de Tendências:**
- `{trends_analysis}` → observações sobre padrões recorrentes

**Referências:**
- `{developer_references}` → lista de arquivos da arch-py skill citados
- `{external_references}` → links externos citados

**Informações do Review:**
- `{review_date}`, `{review_duration}`, `{base_branch}`, `{compare_branch}`

**Notas Finais:**
- `{final_notes}` → observações adicionais, contexto, próximos passos

**Checklist Status:**
- `{full_checklist_status}` → resumo do checklist com ✅ ❌ para cada item

**c) Monte o relatório final:**

Usando o template lido de `assets/report.md`, substitua TODOS os placeholders pelos dados coletados. O relatório final deve:
- Seguir exatamente a estrutura markdown do template
- Ter todos os placeholders substituídos por valores reais
- Incluir a análise de impacto completa (Opção 1)
- Incluir todos os reviews detalhados (Opção 2)
- Incluir todos os resumos e agregações

**IMPORTANTE:** Você está PREENCHENDO o template, não apenas citando-o. O usuário deve ver o relatório completo e formatado.

**d) Salve o relatório:**

```bash
cat > review-output.md << 'EOF'
{todo o relatório formatado}
EOF
```

**e) Informe ao usuário:**

```
✅ Relatório completo salvo em: review-output.md

📊 Resumo:
- {files_reviewed} arquivos revisados
- {total_comments} comentários gerados
- 🔴 {critical} Critical | 🟠 {high} High | 🟡 {medium} Medium | 🟢 {low} Low

{final_recommendation_emoji} Recomendação: {final_recommendation_text}

Arquivo pronto para copy-paste no PR.
```

---

## Critérios de Decisão Final

Use estes critérios para determinar a recomendação final:

**❌ Não Aprovar (Block Merge):**
- 1+ issues **Critical** presentes
- Exemplos: secrets hardcoded, SQL injection, vulnerabilidades de segurança
- **Ação:** Merge deve ser bloqueado até correção

**⚠️ Aprovar com Ressalvas:**
- 0 issues Critical
- 1+ issues **High** presentes
- **Ação:** Pode mergear, mas deve corrigir antes de produção
- Criar tasks/tickets para correção

**✅ Aprovar:**
- 0 issues Critical
- 0 issues High
- Apenas Medium, Low, e/ou Info
- **Ação:** Pode mergear normalmente
- Issues menores podem ser corrigidos posteriormente

**🎉 Aprovação com Elogios:**
- Poucos ou zero issues (apenas Low/Info)
- Código de alta qualidade
- Boas práticas seguidas consistentemente
- **Ação:** Destacar qualidade do trabalho

---

---

## Checklist de Review

Para cada arquivo Python, verificar (consulte `references/checklist.md` para detalhes):

**Code Quality:**
- [ ] Type hints presentes e corretos
- [ ] Nomes descritivos (variáveis, funções, classes)
- [ ] Funções com responsabilidade única
- [ ] Complexidade ciclomática aceitável
- [ ] Imports organizados
- [ ] Constantes no topo ou em config

**Security:**
- [ ] Secrets não hardcodados
- [ ] Validação de dados externos
- [ ] SQL/NoSQL injection prevenido
- [ ] Autenticação/autorização correta

**Performance:**
- [ ] Queries otimizadas (sem N+1)
- [ ] Algoritmos eficientes
- [ ] Memory leaks prevenidos

**Testing:**
- [ ] Testes correspondentes às mudanças
- [ ] Cobertura adequada
- [ ] Casos edge cobertos

**Documentation:**
- [ ] Docstrings em funções públicas
- [ ] Comentários onde necessário
- [ ] README atualizado se aplicável

**Architecture:**
- [ ] Conformidade com padrões do projeto
- [ ] SOLID principles respeitados
- [ ] Acoplamento baixo

---

## Comandos Git Úteis

Consulte `references/git.md` para lista completa. Principais:

```bash
# Ver arquivos modificados
git diff --name-only {base}..{compare}

# Ver diff de arquivo específico
git diff {base}..{compare} -- {arquivo}

# Ver estatísticas
git diff --stat {base}..{compare}

# Ver apenas arquivos Python
git diff --name-only {base}..{compare} | grep '\.py$'

# Ver commits entre branches
git log {base}..{compare} --oneline

# Ver autores das mudanças
git log {base}..{compare} --format='%an' | sort | uniq -c

# Ver contexto maior no diff
git diff -U10 {base}..{compare} -- {arquivo}
```

---

---

## Quando Usar Scripts vs Manual

**Use análise MANUAL quando:**
- Revisar 1-3 arquivos pequenos
- Mudanças simples e diretas
- Preferir controle total do output
- Scripts não disponíveis no ambiente

**Use SCRIPTS quando:**
- Revisar 5+ arquivos
- PRs grandes (>500 linhas)
- Precisar de métricas automáticas
- Quiser detecção automática de padrões
- Precisar de output estruturado (JSON)

**Fluxo híbrido (RECOMENDADO):**
1. Use `analyze_diff.py` para análise inicial e detecção de padrões
2. Revise manualmente seguindo os templates de `assets/`
3. Use `format_output.py` apenas se tiver dados em JSON para compilar

---

## Scripts Disponíveis

### analyze_diff.py

**Propósito:** Analisa git diff e extrai métricas automáticas, detecta padrões problemáticos.

**Uso:**
```bash
# Análise de summary (para Opção 1)
python scripts/analyze_diff.py \
  --base main \
  --compare feature/xyz \
  --format summary

# Análise de arquivo específico (para Opção 2)
python scripts/analyze_diff.py \
  --file src/api/users.py \
  --base main \
  --compare feature/xyz

# Análise completa (para Opção 3)
python scripts/analyze_diff.py \
  --base main \
  --compare feature/xyz \
  --format full \
  --output analysis.json
```

**Inputs:**
- `--base`: branch base para comparação
- `--compare`: branch a ser revisada
- `--file`: (opcional) analisar arquivo específico
- `--format`: `summary` (default), `full`, ou `json`
- `--output`: (opcional) salvar em arquivo JSON

**Outputs (JSON):**
```json
{
  "stats": {
    "total_files": 12,
    "python_files": 8,
    "additions": 567,
    "deletions": 123,
    "net_change": 444
  },
  "files": [
    {
      "path": "src/api/users.py",
      "status": "M",
      "additions": 87,
      "deletions": 12,
      "complexity": "high"
    }
  ],
  "features": [
    {
      "name": "User Authentication API",
      "files": ["src/api/auth.py", "src/services/auth.py"],
      "impact": "high",
      "risk": "medium"
    }
  ],
  "alerts": [
    {
      "type": "secret_hardcoded",
      "file": "src/config.py",
      "line": 42,
      "severity": "critical"
    },
    {
      "type": "n+1_query",
      "file": "src/api/users.py",
      "line": 156,
      "severity": "high"
    }
  ],
  "metrics": {
    "type_hints_coverage": 0.78,
    "docstring_coverage": 0.65,
    "avg_complexity": 12.5
  }
}
```

**Padrões Detectados Automaticamente:**
- Secrets hardcoded (regex: `password|api_key|secret|token = "..."`)
- N+1 queries (loop com query inside)
- SQL injection patterns (string concatenation em queries)
- Missing type hints
- Missing docstrings
- Imports não utilizados
- Print statements (code smell)
- TODOs adicionados

---

### format_output.py

**Propósito:** Formata comentários gerados e compila em `review-output.md`.

**Uso:**
```bash
# Formatar usando template de relatório completo
python scripts/format_output.py \
  --base main \
  --compare feature/xyz \
  --analysis analysis.json \
  --reviews reviews.json \
  --template assets/report.md \
  --output review-output.md \
  --format bitbucket

# Formatar apenas comentários (sem análise de impacto)
python scripts/format_output.py \
  --reviews reviews.json \
  --template assets/comment.md \
  --output review-output.md
```

**Inputs:**
- `--base`: branch base
- `--compare`: branch compare
- `--analysis`: (opcional) JSON da análise de impacto
- `--reviews`: JSON com lista de comentários gerados
- `--template`: template a usar (`assets/report.md`, `assets/summary.md`, ou `assets/comment.md`)
- `--output`: arquivo de saída (default: `review-output.md`)
- `--format`: `bitbucket` (default), `github`, ou `gitlab`

**Formato de reviews.json:**
```json
[
  {
    "file": "src/api/users.py",
    "comments": [
      {
        "number": 1,
        "lines": "42-45",
        "category": "Security",
        "category_emoji": "🔒",
        "severity": "Critical",
        "severity_emoji": "🔴",
        "issue": "Secret hardcoded no código",
        "current_code": "API_KEY = \"sk-abc123...\"",
        "suggested_code": "from pydantic_settings import BaseSettings...",
        "justification": "Credenciais nunca devem estar no código...",
        "references": ["Arch-Py Skill: references/python/configuration.md"]
      }
    ],
    "positive_points": [
      "Type hints completos",
      "Testes com boa cobertura"
    ],
    "summary": {
      "total_comments": 3,
      "by_category": {"Security": 1, "Performance": 1, "Code Quality": 1},
      "max_severity": "Critical",
      "recommendation": "block"
    }
  }
]
```

**Output:**
- Arquivo `review-output.md` formatado e pronto para copy-paste
- Markdown compatível com Bitbucket/GitHub/GitLab
- Links internos funcionais
- Emojis preservados

---

## Estrutura de Arquivos da Skill

```
review-py/
├── SKILL.md                          (este arquivo - workflow principal)
├── references/
│   ├── checklist.md                 (checklist lean mapeado com arch-py skill)
│   ├── templates.md                 (templates e exemplos de comentários)
│   └── git.md                       (comandos git úteis e workflows)
├── scripts/
│   ├── analyze_diff.py              (parser de git diff + detecção de padrões)
│   └── format_output.py             (formatador de output markdown)
└── assets/
    ├── comment.md                   (template de comentário individual)
    ├── summary.md                   (template de análise de impacto)
    └── report.md                    (template de relatório completo)
```

---

## Guia Rápido: Quando Ler Cada Arquivo

### Assets (Templates - LER e PREENCHER)

| Arquivo | Quando Ler | Propósito | Lido Via |
|---------|------------|-----------|----------|
| `assets/comment.md` | Ao gerar cada comentário individual (Opção 2) | Template base com todos placeholders de um comentário | `view assets/comment.md` |
| `assets/summary.md` | Ao gerar análise de impacto (Opção 1) | Template da análise de impacto com métricas e features | `view assets/summary.md` |
| `assets/report.md` | Ao gerar relatório completo (Opção 3) | Template do relatório final consolidado | `view assets/report.md` |

### References (Documentação - LER para CONSULTAR)

| Arquivo | Quando Ler | Propósito | Lido Via |
|---------|------------|-----------|----------|
| `references/checklist.md` | Durante review de arquivo (Opção 2) | Checklist lean com ponteiros para arch-py skill | `view references/checklist.md` |
| `references/templates.md` | Ao gerar comentários (Opção 2) | Exemplos prontos por tipo de issue (SQL Injection, N+1, etc) | `view references/templates.md` |
| `references/git.md` | Quando precisar de comandos git avançados | Workflows git e comandos úteis | `view references/git.md` |

### Scripts (Executáveis - EXECUTAR)

| Script | Quando Executar | Propósito | Como Executar |
|--------|-----------------|-----------|---------------|
| `analyze_diff.py` | Em todas as opções para análise inicial | Detecta padrões, extrai métricas, agrupa features | `python scripts/analyze_diff.py --base {base} --compare {compare}` |
| `format_output.py` | Opcionalmente ao final para compilar | Formata JSON em markdown usando templates | `python scripts/format_output.py --template {template} --output review-output.md` |

### Fluxo Típico de Leitura

**Opção 1 (Análise de Impacto):**
1. Executar comandos git
2. Executar `analyze_diff.py --format summary`
3. **LER** `view assets/summary.md`
4. Preencher placeholders do template
5. Apresentar resultado ao usuário

**Opção 2 (Review por Arquivo):**
1. Listar arquivos Python
2. Para cada arquivo:
   - Executar `git diff`
   - Executar `analyze_diff.py --file {arquivo}`
   - **LER** `view references/checklist.md` para checks
   - Para cada issue:
     - **LER** `view assets/comment.md` para template
     - **CONSULTAR** `view references/templates.md` se precisar de exemplo
     - Preencher placeholders
   - Acumular comentários
3. Salvar tudo em `review-output.md`

**Opção 3 (Relatório Completo):**
1. Executar Opção 1 → guardar em memória
2. Executar Opção 2 para todos arquivos → guardar em memória
3. Executar `analyze_diff.py --format full`
4. **LER** `view assets/report.md`
5. Preencher TODOS placeholders com dados da Opção 1 + Opção 2
6. Salvar relatório final em `review-output.md`

---

**Descrição dos Arquivos:**

**SKILL.md:**
- Workflow completo de code review
- Instruções de uso dos assets e scripts
- Integração com arch-py skill

**references/checklist.md:**
- Checklist lean de review (25 checks)
- Ponteiros para arch-py skill (zero duplicação)
- Severidade típica de cada check

**references/templates.md:**
- Templates específicos por severidade
- Templates por categoria (Security, Performance, etc)
- Exemplos concretos de comentários bem formatados
- Template de pontos positivos e resumos

**references/git.md:**
- Comandos git para comparação de branches
- Análise de mudanças e autores
- Workflows avançados
- Troubleshooting

**scripts/analyze_diff.py:**
- Parse de git diff
- Detecção automática de padrões (secrets, N+1, etc)
- Cálculo de métricas (complexidade, coverage)
- Identificação de features
- Output em JSON

**scripts/format_output.py:**
- Lê templates de assets/
- Preenche placeholders com dados
- Gera review-output.md formatado
- Suporta Bitbucket/GitHub/GitLab

**assets/comment.md:**
- Template base de comentário individual
- Placeholders para todos os campos
- Usado na Opção 2 (Review por Arquivo)

**assets/summary.md:**
- Template de análise de impacto
- Estatísticas, features, priorização
- Usado na Opção 1 (Análise de Impacto)

**assets/report.md:**
- Template de relatório completo
- Combina summary + reviews + métricas
- Usado na Opção 3 (Relatório Completo)

---

## Referências

### Arquivos desta Skill
- [references/checklist.md](references/checklist.md) - Checklist lean de review
- [references/templates.md](references/templates.md) - Templates e exemplos de comentários
- [references/git.md](references/git.md) - Comandos Git e workflows

### Assets (Templates)
- [assets/comment.md](assets/comment.md) - Template de comentário individual
- [assets/summary.md](assets/summary.md) - Template de análise de impacto
- [assets/report.md](assets/report.md) - Template de relatório completo

### Scripts
- [scripts/analyze_diff.py](scripts/analyze_diff.py) - Análise automática de diffs
- [scripts/format_output.py](scripts/format_output.py) - Formatação de output

### Developer Skill (Best Practices Python)
- [../arch-py/SKILL.md](../arch-py/SKILL.md) - Developer skill principal
- [../arch-py/references/python/](../arch-py/references/python/) - Padrões Python
- [../arch-py/references/testing/](../arch-py/references/testing/) - Padrões de testes
- [../arch-py/references/architecture/](../arch-py/references/architecture/) - Arquitetura

### Output Gerado
- `review-output.md` - Arquivo final salvo na raiz do projeto (copy-paste ready)
