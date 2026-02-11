# Code Review Checklist

Checklist de code review para Python. Cada item aponta para a arch-py skill que contém os padrões completos e exemplos.

---

## Como Usar

**Para cada arquivo Python modificado:**

1. Percorra as categorias abaixo sequencialmente
2. Para cada check, consulte a referência indicada na arch-py skill
3. Marque [x] quando item verificado
4. Se encontrar violação, gere comentário citando:
   - O check violado
   - Severidade típica
   - Referência da arch-py skill

**Severidade é indicativa.** Use bom senso baseado no contexto.

---

## 🔒 Security

### [ ] 1. Secrets e Configurações
**Verificar:**
- Sem API keys, tokens, passwords hardcoded
- Configurações vêm de variáveis de ambiente
- Uso de pydantic-settings ou similar

**Severidade típica:** 🔴 Critical  
**Referência:** [Arch-Py - Configuration](../../arch-py/references/python/configuration.md)

---

### [ ] 2. Validação de Input Externo
**Verificar:**
- Dados de APIs, requests, arquivos são validados
- Uso de Pydantic para schemas
- Campos obrigatórios, tipos, validações customizadas

**Severidade típica:** 🟠 High  
**Referência:** [Arch-Py - Pydantic](../../arch-py/references/python/pydantic.md)

---

### [ ] 3. SQL Injection Prevention
**Verificar:**
- Queries parametrizadas (não concatenação de strings)
- Uso de ORM ou queries preparadas
- Sem f-strings em SQL

**Severidade típica:** 🔴 Critical  
**Referência:** OWASP SQL Injection + ORM best practices

---

### [ ] 4. Autenticação e Autorização
**Verificar:**
- Endpoints protegidos quando necessário
- Verificação de ownership/permissions
- Token validation adequada

**Severidade típica:** 🔴 Critical (endpoints públicos) / 🟠 High (internos)  
**Referência:** [Arch-Py - FastAPI Best Practices](../../arch-py/references/fastapi/best-practices.md)

---

### [ ] 5. Dados Sensíveis em Logs
**Verificar:**
- Sem passwords, tokens, PII em logs
- Logging estruturado sem expor dados sensíveis
- Request/response bodies sanitizados

**Severidade típica:** 🔴 Critical  
**Referência:** [Arch-Py - Logging](../../arch-py/references/python/logging.md)

---

## ⚡ Performance

### [ ] 6. N+1 Queries
**Verificar:**
- Loops com queries dentro
- Eager loading de relacionamentos
- Joins em vez de múltiplas queries

**Severidade típica:** 🟠 High  
**Referência:** ORM documentation (SQLAlchemy, Django ORM)

---

### [ ] 7. Algoritmos Eficientes
**Verificar:**
- Complexidade algorítmica (evitar O(n²) ou pior)
- Estruturas de dados apropriadas
- Operações custosas fora de loops

**Severidade típica:** 🟡 Medium / 🟠 High (se em hot path)  
**Referência:** Algoritmos e estruturas de dados básicos

---

### [ ] 8. Resource Management
**Verificar:**
- Context managers para arquivos, conexões, locks
- Sem memory leaks (caches limitados, referências limpas)
- Recursos liberados adequadamente

**Severidade típica:** 🔴 Critical (leaks confirmados) / 🟠 High (suspeitos)  
**Referência:** [Arch-Py - Context Managers](../../arch-py/references/python/context-managers.md)

---

## 🧪 Testing

### [ ] 9. Cobertura de Testes
**Verificar:**
- Código crítico tem testes (auth, pagamento, dados)
- Novos endpoints/features têm testes
- Coverage >60% (geral), >80% (core), 100% (crítico)

**Severidade típica:** 🔴 Critical (código crítico sem testes) / 🟠 High (cobertura <50%)  
**Referência:** [Arch-Py - Pytest](../../arch-py/references/testing/pytest.md)

---

### [ ] 10. Qualidade dos Testes
**Verificar:**
- Testes não frágeis (sem sleep, sem hardcoded IDs/timestamps)
- Casos edge testados
- Assertions específicas e claras

**Severidade típica:** 🟡 Medium  
**Referência:** [Arch-Py - Testing Best Practices](../../arch-py/references/testing/pytest.md)

---

## ⚡ Code Quality

### [ ] 11. Type Hints
**Verificar:**
- Parâmetros de funções tipados
- Retornos de funções tipados
- Variáveis complexas tipadas
- Uso de tipos modernos (list[str] não List[str])

**Severidade típica:** 🟡 Medium (funções privadas) / 🟠 High (APIs públicas)  
**Referência:** [Arch-Py - Type System](../../arch-py/references/python/type-system.md)

---

### [ ] 12. Error Handling
**Verificar:**
- Try/except em operações que podem falhar
- Exceções específicas (não Exception genérico)
- Erros logados adequadamente
- Cleanup em finally ou context managers

**Severidade típica:** 🔴 Critical (operações críticas) / 🟠 High (APIs) / 🟡 Medium (geral)  
**Referência:** [Arch-Py - Error Handling](../../arch-py/references/python/error-handling.md)

---

### [ ] 13. Logging Estruturado
**Verificar:**
- Logs em operações críticas
- Context incluído (user_id, request_id, order_id)
- Níveis apropriados (info/warning/error)
- Structured logging (JSON) preferido

**Severidade típica:** 🟠 High (APIs e serviços) / 🟡 Medium (código interno)  
**Referência:** [Arch-Py - Logging](../../arch-py/references/python/logging.md)

---

### [ ] 14. Docstrings
**Verificar:**
- APIs públicas documentadas
- Funções complexas explicadas
- Parâmetros e retornos descritos
- Exemplos quando necessário

**Severidade típica:** 🟠 High (APIs públicas) / 🟡 Medium (complexas) / 🟢 Low (simples)  
**Referência:** PEP 257 - Docstring Conventions

---

### [ ] 15. Naming
**Verificar:**
- Nomes revelam intenção
- Convenções seguidas (snake_case funções, PascalCase classes)
- Sem abreviações obscuras
- Consistência no módulo

**Severidade típica:** 🟡 Medium (variáveis) / 🟠 High (APIs públicas)  
**Referência:** PEP 8 - Style Guide

---

### [ ] 16. Single Responsibility Principle
**Verificar:**
- Função faz uma coisa só
- <20-30 linhas idealmente
- Pode ser testada isoladamente
- Nome não contém "e" (process_AND_send_AND_update)

**Severidade típica:** 🟡 Medium / 🟠 High (se muito complexo)  
**Referência:** [Arch-Py - Clean Architecture](../../arch-py/references/architecture/clean-architecture.md)

---

### [ ] 17. DRY (Don't Repeat Yourself)
**Verificar:**
- Sem código duplicado
- Lógica repetida extraída para funções
- Patterns identificados e abstraídos

**Severidade típica:** 🟡 Medium  
**Referência:** Princípio DRY

---

### [ ] 18. Complexidade Ciclomática
**Verificar:**
- Decision points razoáveis (<10 ideal, <15 aceitável)
- Ifs/loops aninhados minimizados
- Função pode ser quebrada se muito complexa

**Severidade típica:** 🟡 Medium (>10) / 🟠 High (>15)  
**Ferramenta:** `radon cc --min C`

---

### [ ] 19. Imports Organizados
**Verificar:**
- Ordem: stdlib → third-party → local
- Sem imports não usados
- Sem imports * (star imports)
- Um import por linha

**Severidade típica:** 🟢 Low  
**Ferramenta:** `ruff check --select I` ou `isort`

---

## 🏗️ Architecture

### [ ] 20. Separação de Responsabilidades
**Verificar:**
- Models não têm lógica de negócio
- Controllers/endpoints são finos
- Services contêm lógica
- Repositories isolam acesso a dados

**Severidade típica:** 🟡 Medium / 🟠 High (violação grave)  
**Referência:** [Arch-Py - Clean Architecture](../../arch-py/references/architecture/clean-architecture.md)

---

### [ ] 21. Dependency Injection
**Verificar:**
- Dependências injetadas, não importadas diretamente
- Facilita testing com mocks
- Configurações vêm de fora

**Severidade típica:** 🟡 Medium  
**Referência:** [Arch-Py - Dependency Injection](../../arch-py/references/architecture/dependency-injection.md)

---

## 🔧 Configuration & Dependencies

### [ ] 22. Dependências Versionadas
**Verificar:**
- Versões pinadas (requirements.txt ou poetry.lock)
- Não usa ranges muito largos
- Dependências de dev separadas

**Severidade típica:** 🟠 High (produção) / 🟡 Medium (dev)  
**Referência:** [Arch-Py - Packaging](../../arch-py/references/python/packaging.md)

---

### [ ] 23. Async/Await Usado Corretamente
**Verificar:**
- I/O-bound operations usam async
- Não bloqueia event loop
- Await em operações assíncronas

**Severidade típica:** 🟠 High (se bloqueia event loop) / 🟡 Medium (performance)  
**Referência:** [Arch-Py - Async Patterns](../../arch-py/references/python/async-patterns.md)

---

## 📝 Documentation

### [ ] 24. README Atualizado
**Verificar:**
- Setup instructions refletem mudanças
- Novas dependências documentadas
- Novos endpoints/features descritos

**Severidade típica:** 🟡 Medium (novos projetos) / 🟢 Low (estabelecidos)

---

### [ ] 25. CHANGELOG Atualizado
**Verificar:**
- Breaking changes documentadas
- Novas features listadas
- Formato consistente

**Severidade típica:** 🟢 Low

---

## Resumo Rápido

**Ordem de prioridade durante review:**

1. **Security** (checks 1-5) → Máxima prioridade
2. **Performance** (checks 6-8) → Buscar problemas graves
3. **Testing** (checks 9-10) → Coverage e qualidade
4. **Code Quality** (checks 11-19) → Conformidade com arch-py skill
5. **Architecture** (checks 20-21) → Estrutura do código
6. **Config/Deps** (checks 22-23) → Configurações
7. **Documentation** (checks 24-25) → Docs atualizadas

---

## Ferramentas de Apoio

Algumas verificações podem ser automatizadas:
```bash
# Type checking
mypy src/

# Linting
ruff check .

# Formatting
black --check .

# Security
bandit -r src/

# Complexity
radon cc src/ --min C

# Coverage
pytest --cov=src --cov-report=term-missing

# Imports
ruff check --select I
```

**Referência completa:** [Arch-Py - Tooling](../../arch-py/references/tooling/setup.md)

---

## Notas Importantes

**Este checklist é um guia, não uma regra rígida:**
- Use bom senso baseado no contexto do projeto
- Severidades são indicativas, não absolutas
- Consulte sempre a arch-py skill para padrões detalhados
- Adapte para o contexto (startup vs enterprise, prototipo vs produção)

**Para decisão final de aprovação:**
Consulte a seção "Decisão Final" no SKILL.md principal da review-py.