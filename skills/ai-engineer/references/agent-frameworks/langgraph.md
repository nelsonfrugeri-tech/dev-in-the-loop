# LangGraph - State Machines para Agents

Referência técnica completa de LangGraph: state machines, graph-based agents, e workflows complexos.

---

## O que é LangGraph?

**LangGraph** é uma biblioteca para construir **stateful, graph-based applications** com LLMs.

**Core concepts:**
- **State** = Dados que fluem pelo graph
- **Nodes** = Funções que processam state
- **Edges** = Conexões entre nodes (fixas ou condicionais)
- **Graph** = DAG (directed acyclic graph) ou cyclic graph

**Diferença para LangChain:**
- LangChain = Linear chains (A → B → C)
- LangGraph = Complex graphs com routing, loops, conditionals

**Trade-offs:**
- ✅ **Prós:** Control flow explícito, state visível, debugging mais fácil que LangChain, suporta cycles
- ❌ **Contras:** Mais verboso que chains, curva de aprendizado, overhead inicial

---

## Quando Usar LangGraph

### ✅ Use LangGraph quando:

1. **Complex routing** - Decisões condicionais sobre próximo passo
2. **Multi-step reasoning** - Agent precisa de múltiplas etapas (ReAct, Plan-and-Execute)
3. **Cycles/loops** - Agent pode precisar repetir passos
4. **State management** - Precisa controlar state complexo entre steps
5. **Multi-agent** - Múltiplos agents colaborando
6. **Human-in-the-loop** - Pausar para input humano

### ❌ NÃO use LangGraph quando:

1. **Workflow linear simples** - LangChain chain é suficiente
2. **Single LLM call** - Overhead desnecessário
3. **Protótipo ultra-rápido** - LangGraph requer mais setup

---

## Installation

```bash
pip install langgraph langchain-anthropic langchain-core
```

---

## Core Concepts

### 1. State

**State** = TypedDict que define dados do graph

```python
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph

class AgentState(TypedDict):
    """State para agent."""
    messages: Annotated[list[str], "List of messages"]
    user_query: str
    context: str | None
    final_answer: str | None
    iteration: int
```

**Reducers** = Como state é merged quando múltiplos nodes atualizam

```python
from operator import add

class AgentState(TypedDict):
    # Append to list (default behavior seria substituir)
    messages: Annotated[list[str], add]
    # Replace (default)
    current_step: str
```

### 2. Nodes

**Node** = Função que recebe state, retorna state update

```python
from langchain_anthropic import ChatAnthropic

claude = ChatAnthropic(model="claude-3-5-sonnet-20241022")

async def call_llm(state: AgentState) -> AgentState:
    """Node que chama LLM."""
    query = state["user_query"]

    response = await claude.ainvoke(f"Answer: {query}")

    return {
        "messages": [response.content],
        "final_answer": response.content
    }
```

**Node types:**
- **Regular node:** Retorna partial state update
- **End node:** Não retorna nada (termina execução)

### 3. Edges

**Fixed edges** = Sempre vai de A → B

```python
from langgraph.graph import StateGraph, END

graph = StateGraph(AgentState)

# Add nodes
graph.add_node("start", start_node)
graph.add_node("process", process_node)

# Fixed edges
graph.add_edge("start", "process")  # start sempre vai para process
graph.add_edge("process", END)       # process sempre termina
```

**Conditional edges** = Routing baseado em state

```python
def should_continue(state: AgentState) -> str:
    """Decide próximo node baseado em state."""
    if state["iteration"] >= 5:
        return "end"
    elif state["final_answer"]:
        return "end"
    else:
        return "continue"

graph.add_conditional_edges(
    "process",
    should_continue,
    {
        "continue": "process",  # Loop back
        "end": END
    }
)
```

### 4. Graph Execution

```python
# Set entry point
graph.set_entry_point("start")

# Compile
app = graph.compile()

# Run
result = await app.ainvoke({
    "user_query": "What is RAG?",
    "messages": [],
    "context": None,
    "final_answer": None,
    "iteration": 0
})

print(result["final_answer"])
```

---

## Basic Example: Simple Agent

```python
from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, END
from langchain_anthropic import ChatAnthropic
import asyncio

# State
class SimpleState(TypedDict):
    query: str
    response: str | None

# LLM
claude = ChatAnthropic(model="claude-3-5-sonnet-20241022")

# Nodes
async def process_query(state: SimpleState) -> SimpleState:
    """Process user query."""
    response = await claude.ainvoke(state["query"])
    return {"response": response.content}

# Graph
graph = StateGraph(SimpleState)
graph.add_node("process", process_query)
graph.set_entry_point("process")
graph.add_edge("process", END)

app = graph.compile()

# Run
result = await app.ainvoke({
    "query": "Explain RAG in one sentence",
    "response": None
})

print(result["response"])
```

---

## ReAct Pattern

**ReAct** = Reasoning + Acting loop

```python
from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, END
from langchain_anthropic import ChatAnthropic
from langchain.tools import tool

# Tools
@tool
async def search_docs(query: str) -> str:
    """Search documentation."""
    # Implement actual search
    return f"Documentation about {query}..."

@tool
def calculate(expression: str) -> str:
    """Calculate mathematical expression."""
    try:
        result = eval(expression)  # WARNING: unsafe, demo only
        return str(result)
    except Exception as e:
        return f"Error: {e}"

tools = [search_docs, calculate]
tool_map = {t.name: t for t in tools}

# State
class ReActState(TypedDict):
    messages: Annotated[list, "append"]  # Append messages
    query: str
    thought: str | None
    action: str | None
    action_input: dict | None
    observation: str | None
    final_answer: str | None
    iteration: int

# LLM with tools
claude = ChatAnthropic(
    model="claude-3-5-sonnet-20241022",
    temperature=0
).bind_tools(tools)

# Nodes
async def think(state: ReActState) -> ReActState:
    """Agent thinks and decides action."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Use tools when needed."},
        {"role": "user", "content": state["query"]}
    ]

    # Add previous observations
    if state.get("observation"):
        messages.append({
            "role": "user",
            "content": f"Observation from previous action: {state['observation']}"
        })

    response = await claude.ainvoke(messages)

    # Check if tool use
    if response.tool_calls:
        tool_call = response.tool_calls[0]
        return {
            "action": tool_call["name"],
            "action_input": tool_call["args"],
            "thought": response.content or "Using tool",
            "iteration": state["iteration"] + 1
        }

    # Final answer
    return {
        "final_answer": response.content,
        "thought": "Final answer ready"
    }

async def act(state: ReActState) -> ReActState:
    """Execute action (tool call)."""
    tool_name = state["action"]
    tool_input = state["action_input"]

    # Execute tool
    tool = tool_map[tool_name]
    observation = await tool.ainvoke(tool_input)

    return {"observation": str(observation)}

# Routing
def should_continue(state: ReActState) -> Literal["act", "end"]:
    """Decide if should continue or end."""
    if state.get("final_answer"):
        return "end"
    elif state["iteration"] >= 5:
        return "end"  # Max iterations
    elif state.get("action"):
        return "act"
    else:
        return "end"

# Build graph
graph = StateGraph(ReActState)

graph.add_node("think", think)
graph.add_node("act", act)

graph.set_entry_point("think")

graph.add_conditional_edges(
    "think",
    should_continue,
    {
        "act": "act",
        "end": END
    }
)

graph.add_edge("act", "think")  # After action, think again

app = graph.compile()

# Run
result = await app.ainvoke({
    "query": "What is 25 * 17 + 100? Then search for information about RAG.",
    "messages": [],
    "thought": None,
    "action": None,
    "action_input": None,
    "observation": None,
    "final_answer": None,
    "iteration": 0
})

print(result["final_answer"])
```

---

## Plan-and-Execute Pattern

**Plan** = Cria plano de passos
**Execute** = Executa cada passo
**Replan** = Ajusta plano baseado em resultados

```python
from typing import TypedDict, Annotated, Literal

class PlanExecuteState(TypedDict):
    query: str
    plan: list[str]  # List of steps
    current_step: int
    results: Annotated[list[str], "append"]
    final_answer: str | None

claude = ChatAnthropic(model="claude-3-5-sonnet-20241022")

async def plan(state: PlanExecuteState) -> PlanExecuteState:
    """Create execution plan."""
    prompt = f"""Create a step-by-step plan to answer this query:
{state["query"]}

Return as numbered list of steps."""

    response = await claude.ainvoke(prompt)

    # Parse steps (simplified)
    steps = [
        line.strip()
        for line in response.content.split("\n")
        if line.strip() and line[0].isdigit()
    ]

    return {"plan": steps, "current_step": 0}

async def execute_step(state: PlanExecuteState) -> PlanExecuteState:
    """Execute current step."""
    step = state["plan"][state["current_step"]]

    # Execute step
    prompt = f"""Execute this step:
{step}

Previous results:
{chr(10).join(state["results"]) if state["results"] else "None"}

Provide the result."""

    response = await claude.ainvoke(prompt)

    return {
        "results": [response.content],
        "current_step": state["current_step"] + 1
    }

async def synthesize(state: PlanExecuteState) -> PlanExecuteState:
    """Synthesize final answer from results."""
    prompt = f"""Based on these results, answer the original query:

Query: {state["query"]}

Results:
{chr(10).join(state["results"])}

Final answer:"""

    response = await claude.ainvoke(prompt)

    return {"final_answer": response.content}

def should_continue(state: PlanExecuteState) -> Literal["execute", "synthesize"]:
    """Check if more steps to execute."""
    if state["current_step"] >= len(state["plan"]):
        return "synthesize"
    else:
        return "execute"

# Build graph
graph = StateGraph(PlanExecuteState)

graph.add_node("plan", plan)
graph.add_node("execute", execute_step)
graph.add_node("synthesize", synthesize)

graph.set_entry_point("plan")
graph.add_edge("plan", "execute")

graph.add_conditional_edges(
    "execute",
    should_continue,
    {
        "execute": "execute",      # More steps
        "synthesize": "synthesize"  # Done, synthesize
    }
)

graph.add_edge("synthesize", END)

app = graph.compile()

# Run
result = await app.ainvoke({
    "query": "Research RAG systems and create a summary",
    "plan": [],
    "current_step": 0,
    "results": [],
    "final_answer": None
})

print(result["final_answer"])
```

---

## Checkpointing (Persistence)

**Checkpointing** = Salva state em cada step, permite resume/replay

```python
from langgraph.checkpoint.sqlite import SqliteSaver

# Checkpointer
checkpointer = SqliteSaver.from_conn_string("checkpoints.db")

# Compile with checkpointing
app = graph.compile(checkpointer=checkpointer)

# Run with thread_id (sessão)
config = {"configurable": {"thread_id": "user_123"}}

result = await app.ainvoke(
    {"query": "...", ...},
    config=config
)

# Resume from checkpoint
result2 = await app.ainvoke(
    None,  # Continue from last state
    config=config
)

# Get history
history = await app.aget_state_history(config)
for state in history:
    print(state.values)
```

**Use cases:**
- Resume after error
- Human-in-the-loop (pause, get feedback, continue)
- Debugging (replay from any checkpoint)
- A/B testing (branch from checkpoint)

---

## Human-in-the-Loop

```python
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

class HILState(TypedDict):
    query: str
    draft_answer: str | None
    human_feedback: str | None
    final_answer: str | None

async def draft(state: HILState) -> HILState:
    """Generate draft answer."""
    response = await claude.ainvoke(state["query"])
    return {"draft_answer": response.content}

async def revise(state: HILState) -> HILState:
    """Revise based on human feedback."""
    prompt = f"""Original query: {state["query"]}

Draft answer:
{state["draft_answer"]}

Human feedback:
{state["human_feedback"]}

Revise the answer based on feedback:"""

    response = await claude.ainvoke(prompt)
    return {"final_answer": response.content}

# Graph
graph = StateGraph(HILState)

graph.add_node("draft", draft)
graph.add_node("revise", revise)

graph.set_entry_point("draft")
graph.add_edge("draft", "revise")  # Always go to revise
graph.add_edge("revise", END)

# Compile with checkpointing
checkpointer = SqliteSaver.from_conn_string("hil.db")
app = graph.compile(checkpointer=checkpointer)

# Step 1: Generate draft (pauses before revise)
config = {"configurable": {"thread_id": "session_1"}}

result = await app.ainvoke(
    {
        "query": "Explain RAG",
        "draft_answer": None,
        "human_feedback": None,
        "final_answer": None
    },
    config=config
)

print("Draft:", result["draft_answer"])

# Step 2: Human provides feedback (in real app, from UI)
# Update state manually
state = await app.aget_state(config)
await app.aupdate_state(
    config,
    {"human_feedback": "Add more details about vector databases"}
)

# Step 3: Continue (will revise)
result = await app.ainvoke(None, config=config)
print("Final:", result["final_answer"])
```

---

## Multi-Agent Collaboration

```python
from typing import TypedDict, Annotated, Literal

class MultiAgentState(TypedDict):
    task: str
    research_result: str | None
    code_result: str | None
    review_result: str | None
    final_output: str | None

# Agents (different personas)
researcher = ChatAnthropic(
    model="claude-3-5-sonnet-20241022",
    temperature=0.7
)

coder = ChatAnthropic(
    model="claude-3-5-sonnet-20241022",
    temperature=0
)

reviewer = ChatAnthropic(
    model="claude-3-5-sonnet-20241022",
    temperature=0
)

# Nodes
async def research_node(state: MultiAgentState) -> MultiAgentState:
    """Research agent."""
    response = await researcher.ainvoke(
        f"Research this task and provide key information: {state['task']}"
    )
    return {"research_result": response.content}

async def code_node(state: MultiAgentState) -> MultiAgentState:
    """Coding agent."""
    prompt = f"""Task: {state['task']}

Research:
{state['research_result']}

Write Python code to accomplish this task:"""

    response = await coder.ainvoke(prompt)
    return {"code_result": response.content}

async def review_node(state: MultiAgentState) -> MultiAgentState:
    """Review agent."""
    prompt = f"""Review this code:

{state['code_result']}

Provide feedback and improved version if needed:"""

    response = await reviewer.ainvoke(prompt)
    return {"review_result": response.content}

async def synthesize_node(state: MultiAgentState) -> MultiAgentState:
    """Synthesize results."""
    output = f"""Research: {state['research_result']}

Code: {state['code_result']}

Review: {state['review_result']}"""

    return {"final_output": output}

# Graph
graph = StateGraph(MultiAgentState)

graph.add_node("research", research_node)
graph.add_node("code", code_node)
graph.add_node("review", review_node)
graph.add_node("synthesize", synthesize_node)

# Sequential flow
graph.set_entry_point("research")
graph.add_edge("research", "code")
graph.add_edge("code", "review")
graph.add_edge("review", "synthesize")
graph.add_edge("synthesize", END)

app = graph.compile()

# Run
result = await app.ainvoke({
    "task": "Create a function to parse CSV files efficiently",
    "research_result": None,
    "code_result": None,
    "review_result": None,
    "final_output": None
})

print(result["final_output"])
```

---

## Streaming

**Stream intermediate results:**

```python
async for event in app.astream({
    "query": "Complex task",
    ...
}):
    print(event)
    # Output: node outputs as they happen
```

**Stream with checkpointing:**

```python
config = {"configurable": {"thread_id": "123"}}

async for event in app.astream(
    {"query": "..."},
    config=config
):
    node_name = list(event.keys())[0]
    node_output = event[node_name]
    print(f"Node {node_name}: {node_output}")
```

---

## Best Practices

### 1. Use TypedDict para State

```python
# ❌ Plain dict
state = {"query": "...", "result": None}

# ✅ TypedDict
class State(TypedDict):
    query: str
    result: str | None
```

**Por quê:** Type safety, IDE autocomplete, self-documenting

### 2. Nodes Should Be Pure Functions

```python
# ❌ Side effects
global_var = []

async def node(state):
    global_var.append(state["x"])  # BAD
    return state

# ✅ Pure function
async def node(state: State) -> State:
    return {"result": process(state["input"])}
```

**Por quê:** Testability, debuggability, replay from checkpoints

### 3. Use Conditional Edges para Routing

```python
# ✅ Clear routing logic
def should_continue(state: State) -> Literal["continue", "end"]:
    if state["iteration"] >= max_iter:
        return "end"
    return "continue"

graph.add_conditional_edges("process", should_continue, {
    "continue": "process",
    "end": END
})
```

### 4. Enable Checkpointing em Produção

```python
from langgraph.checkpoint.sqlite import SqliteSaver

checkpointer = SqliteSaver.from_conn_string("prod.db")
app = graph.compile(checkpointer=checkpointer)
```

**Por quê:** Recovery, debugging, human-in-the-loop

### 5. Set Max Iterations

```python
def should_continue(state: State) -> str:
    if state["iteration"] >= 10:  # ✅ Prevent infinite loops
        return "end"
    # ... other logic
```

---

## Common Pitfalls

### 1. Infinite loops

**Problem:** Graph cycles sem stopping condition
```python
# ❌ No stop condition
graph.add_edge("process", "process")
```

**Solution:** Add iteration counter + max limit
```python
def should_continue(state):
    if state["iteration"] >= max_iter:
        return "end"
    return "continue"
```

### 2. State não atualiza

**Problem:** Node não retorna partial update
```python
# ❌ Modifica state in-place
async def node(state):
    state["result"] = "..."  # BAD

# ✅ Return update
async def node(state):
    return {"result": "..."}
```

### 3. Checkpointing overhead

**Problem:** SQLite checkpointer slow para high throughput

**Solution:** Use in-memory checkpointer ou Redis
```python
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()  # Fast, mas não persiste
```

### 4. Complex state merging

**Problem:** State merge behavior não é claro

**Solution:** Use Annotated reducers
```python
from operator import add

class State(TypedDict):
    # Append to list
    messages: Annotated[list, add]
    # Replace value
    current: str
```

---

## Troubleshooting

### Graph não termina

**Debug:**
```python
# Add verbose logging
for event in app.stream(input_data):
    print(f"Node: {event}")

# Check iteration count
print(f"Final state iteration: {result['iteration']}")
```

### State não contém expected keys

**Debug:**
```python
# Print state após cada node
async for event in app.astream(input_data):
    node_name = list(event.keys())[0]
    print(f"{node_name} state: {event[node_name]}")
```

### Checkpoint not saving

**Check:**
- Checkpointer configured: `app = graph.compile(checkpointer=...)`
- Config has thread_id: `config = {"configurable": {"thread_id": "..."}}`
- Database file writable

---

## Comparison: LangChain vs LangGraph

| Feature | LangChain Chains | LangGraph |
|---------|------------------|-----------|
| **Flow** | Linear (A → B → C) | Graph (cycles, conditionals) |
| **State** | Implicit (chain memory) | Explicit (TypedDict) |
| **Routing** | Limited (routers) | Full conditional edges |
| **Cycles** | ❌ Não suporta | ✅ Suporta |
| **Debugging** | Difícil | Mais fácil (state visível) |
| **Checkpointing** | Limited | ✅ Native support |
| **Complexity** | Simples | Médio |
| **Use case** | Linear workflows | Complex agents |

**Decision tree:**
```
Simple linear workflow (RAG, summarization)?
  → LangChain Chain

Complex routing/state?
  → LangGraph

Multi-step agent with loops?
  → LangGraph

Multi-agent collaboration?
  → LangGraph
```

---

## References

### External

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangGraph Tutorials](https://langchain-ai.github.io/langgraph/tutorials/)
- [LangGraph Examples](https://github.com/langchain-ai/langgraph/tree/main/examples)
- [LangGraph API Reference](https://langchain-ai.github.io/langgraph/reference/)

### Internal (ai-engineer skill)

- [LangChain](./langchain.md)
- [Custom Agents](./custom-agents.md)
- [Multi-Agent Systems](./multi-agent.md)
- [Tool Integration](./tool-integration.md)
- [RAG Architecture](../rag/architecture.md)

### Internal (arch-py skill)

- [Async Patterns](../../arch-py/references/python/async-patterns.md)
- [Type System](../../arch-py/references/python/type-system.md)
- [Testing](../../arch-py/references/python/testing.md)
