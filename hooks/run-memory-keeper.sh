#!/bin/bash
export PATH="/Users/nelson.frugeri/.nvm/versions/node/v22.15.0/bin:$PATH"
exec npx --yes mcp-memory-keeper "$@"
