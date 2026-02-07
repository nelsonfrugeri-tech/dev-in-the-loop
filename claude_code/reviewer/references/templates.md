# Comment Templates

Templates de comentários para code review. Use estes templates ao gerar comentários, preenchendo os placeholders indicados.

---

## Template Base (Completo)

Use este template para comentários detalhados:
````markdown
**Linhas:** {start_line}-{end_line}  
**Categoria:** {emoji} {categoria}  
**Severidade:** {emoji} {severidade}

**Issue:**
{descrição clara e objetiva do problema em 1-2 frases}

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
- Developer Skill: [{arquivo}](../developer/{caminho})
{outras referências se aplicável}
````

---

## Templates por Severidade

### 🔴 Critical
````markdown
**Linhas:** {start_line}-{end_line}  
**Categoria:** 🔒 Security  
**Severidade:** 🔴 Critical

**Issue:**
{descrição do problema crítico}

**Código Atual:**
```python
{código problemático}
```

**Código Sugerido:**
```python
{código corrigido}
```

**Justificativa:**
Este é um problema crítico que pode causar {impacto grave}.
{explicação técnica detalhada}

**Impacto:**
- {consequência 1}
- {consequência 2}
- {consequência 3}

**Ação Requerida:** Bloqueia merge. Deve ser corrigido imediatamente.

**Referência:**
- Developer Skill: [{arquivo}](../developer/{caminho})
````

---

### 🟠 High
````markdown
**Linhas:** {start_line}-{end_line}  
**Categoria:** {emoji} {categoria}  
**Severidade:** 🟠 High

**Issue:**
{descrição do problema}

**Código Atual:**
```python
{código problemático}
```

**Código Sugerido:**
```python
{código corrigido}
```

**Justificativa:**
{explicação do problema e impacto}

**Impacto:** {impacto em produção se não corrigir}

**Ação Requerida:** Deve corrigir antes de merge.

**Referência:**
- Developer Skill: [{arquivo}](../developer/{caminho})
````

---

### 🟡 Medium
````markdown
**Linhas:** {start_line}-{end_line}  
**Categoria:** {emoji} {categoria}  
**Severidade:** 🟡 Medium

**Issue:**
{descrição do problema}

**Código Atual:**
```python
{código problemático}
```

**Código Sugerido:**
```python
{código corrigido}
```

**Justificativa:**
{explicação do porquê isso é importante}

**Referência:**
- Developer Skill: [{arquivo}](../developer/{caminho})
````

---

### 🟢 Low
````markdown
**Linhas:** {start_line}-{end_line}  
**Categoria:** {emoji} {categoria}  
**Severidade:** 🟢 Low

**Issue:**
{sugestão de melhoria}

**Código Atual:**
```python
{código atual}
```

**Sugestão:**
```python
{código melhorado}
```

**Benefício:** {pequena melhoria que traz}
````

---

### ℹ️ Info
````markdown
**Linhas:** {start_line}-{end_line}  
**Categoria:** ℹ️ Info

**Observação:**
{informação útil ou contexto adicional}

**Contexto:**
{explicação ou alternativa}

**Referência:** {se aplicável}
````

---

## Templates por Categoria

### 🔒 Security - Secret Hardcoded
````markdown
**Linhas:** {start_line}-{end_line}  
**Categoria:** 🔒 Security  
**Severidade:** 🔴 Critical

**Issue:**
Secret key hardcoded no código. Credenciais nunca devem estar no código fonte.

**Código Atual:**
```python
{código com secret hardcoded}
```

**Código Sugerido:**
```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    {secret_field_name}: str
    
    class Config:
        env_file = ".env"

settings = Settings()
```

**Justificativa:**
- Secrets no código vazam via Git history
- Dificulta rotação de credenciais
- Viola OWASP A02:2021 - Cryptographic Failures
- Qualquer pessoa com acesso ao repositório tem acesso

**Impacto:** Comprometimento total do sistema se credenciais vazarem.

**Ação Requerida:** Bloqueia merge. Corrigir imediatamente e rotacionar credenciais.

**Referência:**
- Developer Skill: [references/python/configuration.md](../developer/references/python/configuration.md)
- OWASP: https://owasp.org/Top10/A02_2021-Cryptographic_Failures/
````

---

### 🔒 Security - SQL Injection
````markdown
**Linhas:** {start_line}-{end_line}  
**Categoria:** 🔒 Security  
**Severidade:** 🔴 Critical

**Issue:**
Vulnerabilidade de SQL Injection. Query está sendo construída por concatenação de strings.

**Código Atual:**
```python
{código com SQL injection}
```

**Código Sugerido:**
```python
# Opção 1: Query parametrizada
query = "SELECT * FROM users WHERE email = :email"
result = db.execute(query, {"email": user_email})

# Opção 2: ORM (preferido)
user = db.query(User).filter_by(email=user_email).first()
```

**Justificativa:**
Atacante pode injetar SQL arbitrário e:
- Ler dados sensíveis de qualquer tabela
- Modificar ou deletar dados
- Escalar privilégios
- Executar comandos no servidor

**Impacto:** Comprometimento total do banco de dados.

**Ação Requerida:** Bloqueia merge. Corrigir imediatamente.

**Referência:**
- OWASP SQL Injection: https://owasp.org/www-community/attacks/SQL_Injection
````

---

### 🔒 Security - Input Validation
````markdown
**Linhas:** {start_line}-{end_line}  
**Categoria:** 🔒 Security  
**Severidade:** 🟠 High

**Issue:**
Input externo não validado. Dados de origem externa devem ser sempre validados.

**Código Atual:**
```python
{código que confia cegamente no input}
```

**Código Sugerido:**
```python
from pydantic import BaseModel, field_validator

class {ModelName}(BaseModel):
    {field_name}: {field_type}
    
    @field_validator("{field_name}")
    @classmethod
    def validate_{field_name}(cls, v: {field_type}) -> {field_type}:
        # validação customizada
        if not {condition}:
            raise ValueError("{error_message}")
        return v
```

**Justificativa:**
Sem validação, dados inválidos podem:
- Causar erros não tratados
- Bypass de regras de negócio
- Corrupção de dados no banco

**Ação Requerida:** Corrigir antes de merge.

**Referência:**
- Developer Skill: [references/python/pydantic.md](../developer/references/python/pydantic.md)
````

---

### ⚡ Performance - N+1 Query
````markdown
**Linhas:** {start_line}-{end_line}  
**Categoria:** ⚡ Performance  
**Severidade:** 🟠 High

**Issue:**
N+1 query detectado. Loop executando query a cada iteração.

**Código Atual:**
```python
{código com N+1}
```

**Código Sugerido:**
```python
# SQLAlchemy - Eager loading
from sqlalchemy.orm import joinedload

{objects} = db.query({Model}).options(
    joinedload({Model}.{relationship})
).all()

# Agora {relationship} já está carregado
for obj in {objects}:
    # usa obj.{relationship} sem query adicional
    pass
```

**Justificativa:**
Performance degrada linearmente com o número de registros.
- 10 registros = 11 queries
- 100 registros = 101 queries
- 1000 registros = 1001 queries

**Impacto:** 
- Lentidão significativa
- Timeouts em produção
- Carga desnecessária no banco

**Ação Requerida:** Corrigir antes de merge.

**Referência:**
- SQLAlchemy Relationship Loading: https://docs.sqlalchemy.org/en/20/orm/queryguide/relationships.html
````

---

### ⚡ Code Quality - Type Hints Missing
````markdown
**Linhas:** {start_line}-{end_line}  
**Categoria:** ⚡ Code Quality  
**Severidade:** 🟡 Medium

**Issue:**
Type hints faltando em função/método.

**Código Atual:**
```python
{código sem type hints}
```

**Código Sugerido:**
```python
{código com type hints}
```

**Justificativa:**
Type hints melhoram:
- Segurança de tipos (detecção de erros em tempo de desenvolvimento)
- Autocomplete em IDEs
- Documentação inline
- Refactoring mais seguro

**Referência:**
- Developer Skill: [references/python/type-system.md](../developer/references/python/type-system.md)
````

---

### ⚡ Code Quality - Error Handling
````markdown
**Linhas:** {start_line}-{end_line}  
**Categoria:** ⚡ Code Quality  
**Severidade:** {🔴 Critical / 🟠 High / 🟡 Medium}

**Issue:**
{descrição do problema de error handling}

**Código Atual:**
```python
{código sem tratamento adequado}
```

**Código Sugerido:**
```python
try:
    {operação}
except {SpecificException} as e:
    logger.error(f"{context}: {e}")
    {tratamento apropriado}
    raise  # ou raise CustomException() from e
```

**Justificativa:**
{explicação do porquê é importante tratar este erro}

**Impacto:** {consequência de não tratar}

**Referência:**
- Developer Skill: [references/python/error-handling.md](../developer/references/python/error-handling.md)
````

---

### ⚡ Code Quality - Logging Missing
````markdown
**Linhas:** {start_line}-{end_line}  
**Categoria:** ⚡ Code Quality  
**Severidade:** 🟠 High

**Issue:**
Falta de logging em operação crítica.

**Código Atual:**
```python
{código sem logging}
```

**Código Sugerido:**
```python
import structlog

logger = structlog.get_logger()

def {function_name}({params}):
    log = logger.bind({context_fields})
    log.info("{operation}_started")
    
    try:
        {operação}
        log.info("{operation}_completed", {result_fields})
    except Exception as e:
        log.error("{operation}_failed", error=str(e))
        raise
```

**Justificativa:**
Logs são essenciais para:
- Debug de problemas em produção
- Auditoria de operações críticas
- Monitoring e alertas
- Rastreamento de requests

**Referência:**
- Developer Skill: [references/python/logging.md](../developer/references/python/logging.md)
````

---

### 🧪 Testing - Missing Tests
````markdown
**Linhas:** {start_line}-{end_line}  
**Categoria:** 🧪 Testing  
**Severidade:** {🔴 Critical / 🟠 High}

**Issue:**
{Código crítico / Nova funcionalidade} sem testes correspondentes.

**Sugestão de Testes:**
```python
import pytest

def test_{function_name}_success():
    # Arrange
    {setup}
    
    # Act
    result = {function_name}({params})
    
    # Assert
    assert {expected_outcome}

def test_{function_name}_error_case():
    with pytest.raises({ExpectedException}):
        {function_name}({invalid_params})

@pytest.mark.parametrize("input,expected", [
    ({case_1}),
    ({case_2}),
    ({case_3}),
])
def test_{function_name}_multiple_cases(input, expected):
    assert {function_name}(input) == expected
```

**Justificativa:**
{Por que este código precisa de testes}

**Coverage Esperada:** {X}% para este módulo

**Referência:**
- Developer Skill: [references/testing/pytest.md](../developer/references/testing/pytest.md)
````

---

### 📝 Documentation - Missing Docstring
````markdown
**Linhas:** {start_line}-{end_line}  
**Categoria:** 📝 Documentation  
**Severidade:** 🟠 High

**Issue:**
Função pública/complexa sem docstring.

**Código Atual:**
```python
{código sem docstring}
```

**Código Sugerido:**
```python
def {function_name}({params}) -> {return_type}:
    """
    {Breve descrição do que a função faz em uma linha}
    
    {Descrição mais detalhada se necessário, explicando lógica complexa,
    edge cases, ou considerações importantes}
    
    Args:
        {param_name}: {descrição do parâmetro}
        {param_name}: {descrição do parâmetro}
        
    Returns:
        {descrição do retorno}
        
    Raises:
        {Exception}: {quando é lançada}
        
    Example:
        >>> {exemplo de uso}
        {resultado esperado}
    """
    {código}
```

**Justificativa:**
APIs públicas e funções complexas precisam de documentação para:
- Outros desenvolvedores saberem como usar
- Evitar uso incorreto
- Facilitar manutenção futura

**Referência:**
- PEP 257: https://peps.python.org/pep-0257/
````

---

## Template de Pontos Positivos

Use sempre ao final do review de cada arquivo:
````markdown
### ✅ Pontos Positivos

1. ✨ {aspecto bem implementado}
2. ✨ {boa prática seguida}
3. ✨ {qualidade destacada}
````

**Exemplos concretos:**
````markdown
### ✅ Pontos Positivos

1. ✨ Type hints completos e corretos em todas as funções
2. ✨ Error handling robusto com exceções específicas
3. ✨ Testes com boa cobertura (87%) incluindo casos edge
4. ✨ Logging estruturado com contexto adequado
5. ✨ Código bem organizado seguindo Single Responsibility Principle
````

---

## Template de Resumo por Arquivo
````markdown
### 📊 Resumo: `{caminho/arquivo.py}`

| Categoria | Count | Severidade Máxima |
|-----------|-------|-------------------|
| 🔒 Security | {n} | {max_severity} |
| ⚡ Performance | {n} | {max_severity} |
| 🧪 Testing | {n} | {max_severity} |
| ⚡ Code Quality | {n} | {max_severity} |
| 📝 Documentation | {n} | {max_severity} |
| **Total** | **{total}** | **{overall_max}** |

**Recomendação:** {✅ Aprovar / ⚠️ Aprovar com ressalvas / ❌ Não aprovar}

**Justificativa:** {razão concisa da recomendação}
````

---

## Template de Issue Simples (One-liner)

Para issues muito simples, use formato compacto:
````markdown
**L{line_num}** - {emoji} {severity} - {issue_description} → Sugestão: {quick_fix}  
Ref: [Developer - {topic}](../developer/references/{path})
````

**Exemplo:**
````markdown
**L42** - 🟢 Low - Variável `count` não usada → Remover ou usar no cálculo  
Ref: [Developer - Code Quality](../developer/references/python/best-practices.md)
````

---

## Placeholders Comuns

**Severidades:**
- `🔴 Critical`
- `🟠 High`
- `🟡 Medium`
- `🟢 Low`
- `ℹ️ Info`

**Categorias:**
- `🔒 Security`
- `⚡ Performance`
- `🧪 Testing`
- `📝 Documentation`
- `⚡ Code Quality`
- `🏗️ Architecture`

**Emojis de Resultado:**
- `✅` - Aprovar
- `⚠️` - Aprovar com ressalvas
- `❌` - Não aprovar
- `🎉` - Aprovação com elogios
- `✨` - Ponto positivo
- `🚫` - Bloqueio

---

## Notas de Uso

**Escolha do template:**
1. Use template completo para issues complexos
2. Use template por severidade para issues padrão
3. Use template por categoria para issues específicos conhecidos
4. Use template one-liner para issues triviais

**Personalização:**
- Sempre adapte o template ao contexto
- Adicione detalhes específicos ao código em questão
- Seja específico sobre linhas afetadas
- Cite a developer skill quando aplicável

**Formato Bitbucket:**
- Markdown padrão funciona
- Code blocks com ```python funcionam
- Links internos funcionam
- Emojis funcionam