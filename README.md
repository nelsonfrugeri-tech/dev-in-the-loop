# claude-code

> Create software with AI coders — um sistema multi-agent para o Claude Code que transforma o diretório `~/.claude` em um ambiente de desenvolvimento inteligente com agentes especializados e base de conhecimento versionada.

## O que é este projeto?

O **claude-code** é uma coleção de **agents** e **skills** projetada para ser instalada no diretório `~/.claude` e utilizada com o [Claude Code](https://docs.anthropic.com/en/docs/claude-code) da Anthropic. Ele define um ecossistema de agentes de IA especializados que colaboram entre si para cobrir todo o ciclo de vida do desenvolvimento de software — desde a análise de um repositório até a implementação de código, code review, debate técnico e deploy de infraestrutura local.

A ideia central é que cada agente tem um papel claro e delimitado, consome e produz artefatos padronizados (como `context.md` e issues em Markdown), e pode invocar outros agentes quando necessário, formando pipelines multi-agent.

## Arquitetura

O projeto se organiza em dois conceitos fundamentais:

**Agents** são definições de comportamento para o Claude Code. Cada arquivo `.md` dentro de `agents/` descreve o papel, personalidade, workflow passo a passo, ferramentas permitidas e padrões de comunicação de um agente. Eles são o "quem faz o quê".

**Skills** são bases de conhecimento técnico que os agentes consultam como referência. Cada skill contém um `SKILL.md` descritivo e uma pasta `references/` com materiais de apoio sobre tópicos específicos. Elas são o "como fazer bem feito".

```
~/.claude/
├── agents/                    # Agents especializados
│   ├── builder.md             #   Sobe infraestrutura local
│   ├── debater.md             #   Debate e melhoria de skills
│   ├── dev-py.md              #   Desenvolvimento Python
│   ├── executor.md            #   Implementa melhorias de skills
│   ├── explorer.md            #   Análise de repositórios
│   └── review-py.md           #   Code review Python
│
├── skills/                    # Knowledge bases reutilizáveis
│   ├── ai-engineer/           #   AI/ML engineering (LLM, RAG, Agents)
│   ├── arch-py/               #   Arquitetura Python
│   └── review-py/             #   Templates e critérios de review
│
├── hooks/                     # Automações do Claude Code
│   ├── memory-keeper-save.sh  #   Auto-save de contexto (PreCompact/Stop)
│   ├── memory-keeper-restore.sh # Auto-restore de contexto (SessionStart)
│   ├── memory-keeper-purge.sh #   Expurgo periódico (cron: 15/7 dias)
│   ├── run-memory-keeper.sh   #   Wrapper MCP (gerado pelo bootstrap)
│   ├── setup-cron.sh          #   Instalador do cron de expurgo
│   └── logs/                  #   Logs (não versionado)
│
├── setup/                     # Onboarding e configuração
│   ├── mcp-manifest.json      #   MCPs necessários (declarativo)
│   ├── bootstrap.sh           #   Script de setup por máquina
│   ├── .local-state.json      #   Estado local (não versionado)
│   └── bootstrap.log          #   Log do bootstrap (não versionado)
│
├── CLAUDE.md                  # Instruções globais para o Claude
├── settings.json              # Settings compartilhados (hooks, env vars)
└── .gitignore
```

### O que NÃO é versionado

| Arquivo | Local | Motivo |
|---------|-------|--------|
| `~/.claude.json` | `~/` | MCPs user-scoped, específico da máquina |
| `~/mcp-data/` | `~/` | Dados do Memory Keeper (SQLite) |
| `hooks/logs/` | repo | Logs de execução |
| `setup/.local-state.json` | repo | Estado do bootstrap da máquina |
| `workspace/` | repo | Outputs dos agents por projeto |
| `projects/` | repo | Sessões e memória do Claude Code |

## Agents

### Explorer

O ponto de partida recomendado para qualquer pipeline. Analisa profundamente um repositório e gera um relatório estruturado `context.md` em `.claude/workspace/{projeto}/`. O relatório cobre identidade do projeto, arquitetura, contratos de serviço (endpoints, workers, CLI), infraestrutura, variáveis de ambiente, análise de qualidade cruzada com a skill `arch-py`, saúde das dependências e atividade recente. Opera em dois modos: análise completa (primeira execução) e atualização incremental (execuções subsequentes baseadas no delta de commits).

**Trigger:** `/explorer`
**Skill base:** `arch-py`
**Modelo:** Opus

### Dev-Py

Agente de desenvolvimento Python com personalidade questionadora e obsessão por qualidade. Segue um workflow rígido de 8 etapas: questionar (entender o problema), pesquisar (buscar referências na web e na documentação), projetar (apresentar opções com trade-offs), testar (test-first, sempre antes de implementar), implementar (código com type hints, error handling, docstrings), validar (mypy, ruff, pytest, coverage), revisar (auto-review contra `arch-py`) e documentar decisões técnicas. Consome o `context.md` do explorer quando disponível.

**Trigger:** qualquer tarefa de implementação Python
**Skill base:** `arch-py`
**Modelo:** Opus

### Review-Py

Agente de code review sistemático entre branches Git. Oferece três modos: análise de impacto (estatísticas e features identificadas), review por arquivo (comentários detalhados com severidade, código atual vs. sugerido, referências) e relatório completo combinando ambos. Os comentários são formatados em Markdown para copy-paste direto em PRs. Usa a skill `review-py` para templates e critérios de severidade, e a skill `arch-py` para avaliar qualidade técnica.

**Trigger:** `/review`
**Skills base:** `review-py`, `arch-py`
**Modelo:** Opus

### Builder

Agente que sobe toda a infraestrutura local de um projeto automaticamente. Lê o `context.md`, identifica dependências (MongoDB, Redis, PostgreSQL), sobe containers Docker, verifica e cria `.env`, instala dependências do projeto, inicia API/frontend e valida tudo com testes de conexão e curl. Inclui tratamento robusto de erros (porta em uso, Docker parado, falha de instalação) e oferece watch mode para monitoramento contínuo.

**Trigger:** `/builder`, `/build`
**Skill base:** `arch-py`

### Debater

Agente debatedor com personalidade configurável (Socrático, Expert, Colaborativo — ou combinações) e profundidade ajustável. Debate sobre tópicos das skills, pesquisa estado da arte via web, analisa gaps e conteúdo desatualizado, e ao final do debate cria issues estruturadas em Markdown com propostas de melhoria. A personalidade pode ser alterada durante a conversa. Funciona como o motor de melhoria contínua das skills.

**Trigger:** `/debater`, `/debate`
**Skills base:** todas as skills disponíveis

### Executor

Agente que implementa melhorias nas skills a partir de issues criadas pelo Debater. Lista issues disponíveis, lê a issue escolhida, planeja as mudanças, pede aprovação, executa edições no arquivo da skill, valida as modificações e remove a issue automaticamente após sucesso. Só remove a issue se a validação for 100% bem-sucedida.

**Trigger:** `/executor`, `/executar`
**Skills base:** `arch-py`, `review-py`, `ai-engineer`

## Skills

### arch-py

Base de conhecimento sobre arquitetura e padrões Python modernos. Cobre type system, async/await, dataclasses, context managers, decorators, Pydantic v2, error handling, logging, configuration, concurrency, clean architecture, dependency injection e repository pattern. É a skill mais referenciada — usada como baseline de qualidade pelo Explorer, Dev-Py e Review-Py.

### review-py

Contém templates de comentários para code review, checklists, critérios de severidade, templates de relatórios e scripts auxiliares de análise de diff. Usada pelo agente Review-Py para padronizar o formato e a qualidade dos reviews.

### ai-engineer

Base de conhecimento sobre engenharia de AI/ML, incluindo LLMs, RAG e sistemas de agentes. Usada como referência pelo Debater e Executor quando o debate envolve tópicos de inteligência artificial.

## Fluxo Multi-Agent

Os agentes são projetados para trabalhar em pipeline. O fluxo mais comum é:

1. **Explorer** analisa o repositório e gera `context.md`
2. **Dev-Py** consome o `context.md` e implementa features/bug fixes com qualidade
3. **Review-Py** consome o `context.md` e faz code review entre branches
4. **Builder** consome o `context.md` e sobe toda infra local

Para melhoria contínua das próprias skills:

1. **Debater** debate um tópico e cria issues com propostas de melhoria
2. **Executor** lê as issues, implementa as mudanças nas skills e remove as issues

## Setup (primeira vez)

### Cenário A: Máquina nova (sem `~/.claude`)

```bash
REPO_URL=git@github.com:<org>/dotclaude.git \
  bash <(curl -sL https://raw.githubusercontent.com/<org>/dotclaude/main/setup/bootstrap.sh) --init
```

### Cenário B: Máquina com Claude Code já instalado (`~/.claude` existe)

```bash
# 1. Baixe o bootstrap (pode ser via curl ou copiar manualmente)
curl -sL https://raw.githubusercontent.com/<org>/dotclaude/main/setup/bootstrap.sh \
  -o /tmp/bootstrap.sh

# 2. Rode o init (faz backup automático antes de sincronizar)
REPO_URL=git@github.com:<org>/dotclaude.git bash /tmp/bootstrap.sh --init

# 3. Instale os MCPs
bash ~/.claude/setup/bootstrap.sh

# 4. Instale o cron de expurgo do Memory Keeper
bash ~/.claude/hooks/setup-cron.sh
```

> O `--init` faz backup de tudo em `~/.claude-backup-<timestamp>/` antes de
> sincronizar. Verifique com `cd ~/.claude && git status` e remova o backup
> quando estiver confortável.

### Verificação

```bash
bash ~/.claude/setup/bootstrap.sh --check
```

## Atualização (repo remoto mudou)

Quando alguém do time fizer push de mudanças (novos agents, skills, hooks):

```bash
# 1. Pull das mudanças
cd ~/.claude && git pull

# 2. Se houve mudança em hooks ou setup/mcp-manifest.json, reinstale MCPs
bash ~/.claude/setup/bootstrap.sh

# 3. Se houve mudança no cron/purge, reinstale o cron
bash ~/.claude/hooks/setup-cron.sh
```

### O que requer ação após pull

| O que mudou | Ação necessária |
|-------------|-----------------|
| `agents/*.md` | Nenhuma (carregado automaticamente) |
| `skills/**` | Nenhuma (carregado automaticamente) |
| `CLAUDE.md` | Nenhuma (carregado automaticamente) |
| `settings.json` | Reiniciar sessão do Claude Code |
| `hooks/*.sh` | Nenhuma (executado automaticamente) |
| `setup/mcp-manifest.json` | `bash ~/.claude/setup/bootstrap.sh` |

### Quick update (one-liner)

```bash
cd ~/.claude && git pull && bash setup/bootstrap.sh && echo "Atualizado!"
```

## Configuração por máquina (MCPs)

MCPs são **específicos por máquina** e ficam em `~/.claude.json` (fora do repo).

O `setup/mcp-manifest.json` declara quais MCPs são necessários e o `bootstrap.sh`
instala automaticamente, detectando o Node disponível na máquina (nvm, fnm ou sistema).

Para adicionar um MCP **só na sua máquina** (sem afetar o time):

```bash
claude mcp add --scope user meu-mcp -- <comando>
```

Para propor um MCP **para todo o time**, adicione no `mcp-manifest.json` e abra um PR.

## Memory Keeper

Sistema de memória persistente entre sessões do Claude Code.

### Como funciona

- **SessionStart** → hook injeta instrução para restaurar contexto
- **PreCompact/Stop** → hook injeta instrução para salvar contexto
- Dados salvos em `~/mcp-data/memory-keeper/` (SQLite, por máquina)

### Política de expurgo

- Cron roda **diariamente** às 3h
- A cada **15 dias**, limpa registros com mais de **7 dias**
- Faz **backup** do SQLite antes de purgar (mantém últimos 3)
- Logs em `~/.claude/hooks/logs/purge.log`

### Comandos manuais

```bash
# Preview do que seria purgado (sem deletar)
~/.claude/hooks/memory-keeper-purge.sh --dry-run

# Forçar expurgo agora
~/.claude/hooks/memory-keeper-purge.sh --force

# Limpar MCPs instalados pelo bootstrap
bash ~/.claude/setup/bootstrap.sh --clean
```

## Contribuindo

1. Crie uma branch: `git checkout -b feat/minha-feature`
2. Faça suas mudanças em agents/, skills/, hooks/ ou setup/
3. Teste localmente
4. Abra PR para `main`

### Convenções

- **Agents**: um arquivo `.md` por agent em `agents/`
- **Skills**: pasta com `SKILL.md` + `references/` em `skills/<nome>/`
- **Hooks**: scripts `.sh` executáveis em `hooks/`
- **Issues de skills**: criadas pelo agent debater em `issues/skills/<nome>/`

## Licença

Este é um projeto open source. Consulte o repositório para informações sobre licença.
