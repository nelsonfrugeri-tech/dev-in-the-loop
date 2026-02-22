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
├── agents/
│   ├── builder.md        # Infraestrutura local
│   ├── debater.md        # Debate técnico e melhoria contínua
│   ├── dev-py.md         # Desenvolvimento Python
│   ├── executor.md       # Implementação de issues
│   ├── explorer.md       # Análise de repositório
│   └── review-py.md      # Code review Python
├── skills/
│   ├── ai-engineer/      # Conhecimento de AI/ML (LLM, RAG, Agents)
│   │   ├── SKILL.md
│   │   └── references/
│   ├── arch-py/          # Arquitetura e padrões Python modernos
│   │   ├── SKILL.md
│   │   └── references/python/
│   └── review-py/        # Templates e critérios de code review
│       ├── SKILL.md
│       ├── assets/
│       ├── references/
│       └── scripts/
└── .gitignore
```
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
## Instalação
Clone o repositório diretamente no diretório `~/.claude`:
```bash
cd ~/.claude
git clone https://github.com/nelsonfrugeri-tech/claude-code.git .
```
Ou, se `~/.claude` já existe, clone e copie os diretórios:
```bash
git clone https://github.com/nelsonfrugeri-tech/claude-code.git /tmp/claude-code
cp -r /tmp/claude-code/agents ~/.claude/agents
cp -r /tmp/claude-code/skills ~/.claude/skills
```
Os agents e skills estarão automaticamente disponíveis para o Claude Code.
## Uso
Com o Claude Code ativo, invoque qualquer agente pelo trigger:
```
/explorer          → Analisa o repositório atual
/builder           → Sobe infraestrutura local
/debater           → Inicia debate técnico sobre uma skill
/executor          → Lista e implementa issues
```
Para desenvolvimento e code review, os agentes `dev-py` e `review-py` são invocados contextualmente ao trabalhar com projetos Python.
## Convenções do Projeto
O repositório versiona apenas `agents/` e `skills/` (controlado pelo `.gitignore`). Diretórios como `workspace/` (onde ficam os `context.md` gerados) e `issues/` (onde ficam as issues do Debater) são gerados em runtime e não versionados. A linguagem primária dos agents e skills é português brasileiro, e todo o código de referência nas skills é Python.
## Licença
Este é um projeto open source. Consulte o repositório para informações sobre licença.
