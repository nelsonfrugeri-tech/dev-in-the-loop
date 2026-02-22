# Claude Code - Instruções Globais

## Workspace de Projetos

Todos os arquivos gerados por agents/skills (relatórios, context.md, análises, etc.) devem ser salvos em:

```
$CLAUDE_WORKSPACE/<nome_projeto>/
```

Onde:
- `$CLAUDE_WORKSPACE` = `~/.claude/workspace`
- `<nome_projeto>` = basename do git root do projeto atual (`basename $(git rev-parse --show-toplevel 2>/dev/null) || basename $PWD`)

**NUNCA** salve arquivos gerados na raiz do repositório do projeto. Isso evita que arquivos de workspace sejam commitados acidentalmente no repo remoto.

Exemplo: se o projeto está em `/Users/nelson.frugeri/projects/lm-gateway`, salve em `~/.claude/workspace/lm-gateway/`.
