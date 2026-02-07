# Análise de Impacto - Code Review

**Branches:** `{base_branch}` → `{compare_branch}`  
**Data:** {review_date}  
**Reviewer:** Claude (review-py skill)

---

## 📊 Estatísticas Gerais

| Métrica | Valor |
|---------|-------|
| **Commits** | {total_commits} |
| **Arquivos modificados** | {total_files} |
| **Arquivos Python** | {python_files} |
| **Linhas adicionadas** | +{lines_added} |
| **Linhas removidas** | -{lines_removed} |
| **Mudança líquida** | {net_change} |

---

## 📁 Arquivos por Categoria

### Arquivos Python Modificados ({python_modified_count})
{python_modified_list}

### Arquivos Python Adicionados ({python_added_count})
{python_added_list}

### Arquivos Python Removidos ({python_deleted_count})
{python_deleted_list}

### Arquivos Python Renomeados ({python_renamed_count})
{python_renamed_list}

### Outros Arquivos ({other_files_count})
{other_files_list}

---

## 🎯 Features/Mudanças Principais

{features_list}

---

## 👥 Autores das Mudanças

{authors_list}

---

## 📈 Análise de Complexidade

| Arquivo | Linhas +/- | Complexidade |
|---------|------------|--------------|
{complexity_table}

**Legenda de Complexidade:**
- 🟢 **Baixa:** < 50 linhas modificadas
- 🟡 **Média:** 50-200 linhas modificadas
- 🟠 **Alta:** 200-500 linhas modificadas
- 🔴 **Muito Alta:** > 500 linhas modificadas

---

## ⚠️ Alertas Preliminares

{preliminary_alerts}

---

## 🎯 Recomendações de Prioridade

### Alta Prioridade
{high_priority_files}

### Média Prioridade
{medium_priority_files}

### Baixa Prioridade
{low_priority_files}

---

## 📝 Próximos Passos

1. {next_step_1}
2. {next_step_2}
3. {next_step_3}

---

**Nota:** Esta é apenas uma análise de impacto. Para review detalhado, execute a opção "Review por Arquivo" ou "Relatório Completo".