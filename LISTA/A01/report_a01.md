# LiteLLM and LangGraph: Complete Guide and Comparison

## Table of Contents
1. [LiteLLM Comprehensive Tutorial](#litellm-comprehensive-tutorial)
2. [LangGraph Comprehensive Tutorial](#langgraph-comprehensive-tutorial)
3. [Detailed Comparison Analysis](#detailed-comparison-analysis)
4. [Integration Patterns](#integration-patterns)
5. [Decision Framework](#decision-framework)
6. [Production Considerations](#production-considerations)
7. [Key Takeaway](#key-takeaways)
---

## LiteLLM Comprehensive Tutorial

### What is LiteLLM?

LiteLLM is a unified API gateway that standardizes interactions with multiple LLM providers through a single interface. It acts as an abstraction layer that eliminates the complexity of managing different API formats, authentication methods, and response structures across various LLM services.

### Core Architecture

```
Application Code
       ↓
   LiteLLM API
       ↓
Multiple Providers (OpenAI, Anthropic, Azure, Cohere, etc.)
```

### Installation and Setup

```bash
pip install litellm
```

### Basic Usage Examples

#### 1. Simple Model Calls

```python
from litellm import completion
import os

# Set API keys
os.environ["OPENAI_API_KEY"] = "your-openai-key"
os.environ["ANTHROPIC_API_KEY"] = "your-anthropic-key"

# OpenAI GPT-4
response_openai = completion(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Explain quantum computing in simple terms"}]
)
print("OpenAI:", response_openai.choices[0].message.content)

# Anthropic Claude
response_claude = completion(
    model="claude-3-sonnet-20240229",
    messages=[{"role": "user", "content": "Explain quantum computing in simple terms"}]
)
print("Claude:", response_claude.choices[0].message.content)
```

#### 2. Advanced Configuration

```python
from litellm import completion
import litellm

# Configure global settings
litellm.set_verbose = True  # Enable detailed logging
litellm.drop_params = True  # Drop unsupported parameters

# Custom configuration
response = completion(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Write a Python function"}],
    temperature=0.7,
    max_tokens=500,
    top_p=0.9,
    frequency_penalty=0.1,
    presence_penalty=0.1
)
```

### Advanced Features

#### 1. Streaming Responses

```python
from litellm import completion

def stream_example():
    response = completion(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Write a story about AI"}],
        stream=True
    )
    
    for chunk in response:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end='')

stream_example()
```

#### 2. Function Calling

```python
from litellm import completion

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"}
                },
                "required": ["location"]
            }
        }
    }
]

response = completion(
    model="gpt-4o",
    messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
    tools=tools,
    tool_choice="auto"
)

print(response.choices[0].message.tool_calls)
```

#### 3. Cost Tracking and Monitoring

```python
from litellm import completion, cost_per_token
import litellm

# Enable cost tracking
litellm.success_callback = ["langfuse", "lunary"]

def track_costs():
    response = completion(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Analyze this data"}],
        stream=True
    )
    
    total_cost = 0
    for chunk in response:
        if hasattr(chunk, 'usage') and chunk.usage:
            cost = cost_per_token(
                model="gpt-4o",
                prompt_tokens=chunk.usage.prompt_tokens,
                completion_tokens=chunk.usage.completion_tokens
            )
            total_cost += cost
    
    print(f"Total cost: ${total_cost:.4f}")

track_costs()
```

#### 4. Fallback and Retry Logic

```python
from litellm import completion
import litellm

# Configure fallbacks
litellm.set_verbose = True

def robust_completion(messages, models=None):
    if models is None:
        models = ["gpt-4o", "claude-3-sonnet-20240229", "gpt-3.5-turbo"]
    
    for model in models:
        try:
            response = completion(
                model=model,
                messages=messages,
                timeout=30,
                max_retries=3
            )
            return response, model
        except Exception as e:
            print(f"Error with {model}: {str(e)}")
            continue
    
    raise Exception("All models failed")

# Usage
messages = [{"role": "user", "content": "Explain machine learning"}]
response, used_model = robust_completion(messages)
print(f"Used model: {used_model}")
print(response.choices[0].message.content)
```

### Production Setup

#### 1. LiteLLM Proxy Server

```python
# config.yaml
model_list:
  - model_name: gpt-4o
    litellm_params:
      model: openai/gpt-4o
      api_key: os.environ/OPENAI_API_KEY
  - model_name: claude-3-sonnet
    litellm_params:
      model: anthropic/claude-3-sonnet-20240229
      api_key: os.environ/ANTHROPIC_API_KEY
  - model_name: local-llama
    litellm_params:
      model: ollama/llama3
      api_base: http://localhost:11434

general_settings:
  master_key: your-master-key
  database_url: "postgresql://user:password@localhost:5432/litellm"
```

```bash
# Start proxy server
litellm --config config.yaml --port 4000
```

#### 2. Client Usage with Proxy

```python
from litellm import completion

# Use proxy endpoint
response = completion(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello"}],
    api_base="http://localhost:4000",
    api_key="your-master-key"
)
```

### LiteLLM Use Cases

1. **Multi-Provider Applications**: Applications that need to switch between different LLM providers
2. **Cost Optimization**: Routing requests to the most cost-effective provider
3. **Reliability**: Implementing fallback mechanisms for high availability
4. **Standardization**: Unifying LLM interactions across an organization
5. **Development**: Testing applications with multiple models without code changes

---

## LangGraph Comprehensive Tutorial

### What is LangGraph?

LangGraph is a framework for building stateful, multi-actor applications with LLMs. It models applications as directed graphs where nodes represent computation steps and edges define execution flow, enabling complex agent workflows and multi-step reasoning systems.

### Core Concepts

#### 1. Graph Structure
- **Nodes**: Computation steps (LLM calls, tool executions, Python functions)
- **Edges**: Flow control (conditional routing, loops, parallel execution)
- **State**: Shared data structure that persists across nodes

#### 2. Execution Models
- **Synchronous**: Sequential execution
- **Asynchronous**: Concurrent execution for better performance
- **Streaming**: Real-time processing of intermediate results

### Installation and Setup

```bash
pip install langgraph langchain langchain-openai
```

### Basic Examples

#### 1. Simple Chatbot

```python
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
import os

os.environ["OPENAI_API_KEY"] = "your-openai-key"

# Define state structure
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# Define node function
def chatbot_node(state: ChatState):
    llm = ChatOpenAI(model="gpt-4o-mini")
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

# Build graph
graph_builder = StateGraph(ChatState)
graph_builder.add_node("chatbot", chatbot_node)
graph_builder.set_entry_point("chatbot")
graph_builder.add_edge("chatbot", END)

# Compile and run
app = graph_builder.compile()
initial_input = {"messages": [HumanMessage(content="Explain LangGraph")]}
final_state = app.invoke(initial_input)

for message in final_state["messages"]:
    print(f"[{message.type.upper()}]: {message.content}")
```

#### 2. ReAct Agent Pattern

```python
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from typing import TypedDict, Annotated, List
import operator

# Define tools
@tool
def search_tool(query: str) -> str:
    """Search for information about a query."""
    # Simulated search
    return f"Search results for: {query}"

@tool
def calculator_tool(expression: str) -> str:
    """Calculate mathematical expressions."""
    try:
        result = eval(expression)
        return str(result)
    except:
        return "Invalid expression"

tools = [search_tool, calculator_tool]
tool_executor = ToolExecutor(tools)

# Define agent state
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    agent_outcome: str

# Define agent node
def agent_node(state):
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    llm_with_tools = llm.bind_tools(tools)
    
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

# Define tool node
def tool_node(state):
    last_message = state["messages"][-1]
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        tool_call = last_message.tool_calls[0]
        action = tool_call["name"]
        tool_input = tool_call["args"]
        
        # Execute tool
        response = tool_executor.invoke({"tool_name": action, "tool_input": tool_input})
        function_message = HumanMessage(content=str(response))
        return {"messages": [function_message]}
    return {"messages": []}

# Define conditional logic
def should_continue(state):
    last_message = state["messages"][-1]
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    return "end"

# Build graph
workflow = StateGraph(AgentState)
workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)

workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", should_continue, {
    "tools": "tools",
    "end": END
})
workflow.add_edge("tools", "agent")

app = workflow.compile()

# Test the agent
inputs = {"messages": [HumanMessage(content="What is 25 * 4 + 10?")]}
result = app.invoke(inputs)

for message in result["messages"]:
    print(f"[{type(message).__name__}]: {message.content}")
```

#### 3. Multi-Agent System

```python
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from typing import TypedDict, List, Literal
import operator

# Define shared state
class MultiAgentState(TypedDict):
    messages: List[BaseMessage]
    current_agent: str
    task_complete: bool

# Define specialist agents
def researcher_agent(state):
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    system_message = "You are a research specialist. Gather and analyze information."
    
    messages = [HumanMessage(content=system_message)] + state["messages"]
    response = llm.invoke(messages)
    
    return {
        "messages": [AIMessage(content=f"[RESEARCHER]: {response.content}")],
        "current_agent": "writer"
    }

def writer_agent(state):
    llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
    system_message = "You are a writing specialist. Create clear, engaging content."
    
    messages = [HumanMessage(content=system_message)] + state["messages"]
    response = llm.invoke(messages)
    
    return {
        "messages": [AIMessage(content=f"[WRITER]: {response.content}")],
        "current_agent": "reviewer"
    }

def reviewer_agent(state):
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    system_message = "You are a quality reviewer. Check for accuracy and clarity."
    
    messages = [HumanMessage(content=system_message)] + state["messages"]
    response = llm.invoke(messages)
    
    return {
        "messages": [AIMessage(content=f"[REVIEWER]: {response.content}")],
        "task_complete": True
    }

# Define routing logic
def route_agent(state):
    if state.get("current_agent") == "writer":
        return "writer"
    elif state.get("current_agent") == "reviewer":
        return "reviewer"
    elif state.get("task_complete"):
        return "end"
    return "researcher"

# Build multi-agent workflow
workflow = StateGraph(MultiAgentState)
workflow.add_node("researcher", researcher_agent)
workflow.add_node("writer", writer_agent)
workflow.add_node("reviewer", reviewer_agent)

workflow.set_entry_point("researcher")
workflow.add_conditional_edges("researcher", route_agent, {
    "writer": "writer",
    "reviewer": "reviewer",
    "end": END
})
workflow.add_conditional_edges("writer", route_agent, {
    "reviewer": "reviewer",
    "end": END
})
workflow.add_edge("reviewer", END)

app = workflow.compile()

# Test multi-agent system
inputs = {
    "messages": [HumanMessage(content="Write a brief article about renewable energy")],
    "current_agent": "researcher",
    "task_complete": False
}

result = app.invoke(inputs)
for message in result["messages"]:
    print(message.content)
```

### Advanced Features

#### 1. Checkpointing and Persistence

```python
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph, END
import sqlite3

# Create checkpoint saver
memory = SqliteSaver(sqlite3.connect("checkpoints.db", check_same_thread=False))

# Compile with checkpointing
app = workflow.compile(checkpointer=memory)

# Run with thread ID for persistence
config = {"configurable": {"thread_id": "conversation_1"}}
result = app.invoke(inputs, config=config)

# Resume from checkpoint
resumed_result = app.invoke({"messages": [HumanMessage(content="Continue...")]}, config=config)
```

#### 2. Streaming and Real-time Processing

```python
async def stream_example():
    inputs = {"messages": [HumanMessage(content="Explain machine learning")]}
    
    async for chunk in app.astream(inputs):
        print(f"Chunk: {chunk}")
        
    # Stream with events
    async for event in app.astream_events(inputs, version="v1"):
        print(f"Event: {event}")

# Run async
import asyncio
asyncio.run(stream_example())
```

#### 3. Human-in-the-Loop

```python
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

def human_approval_node(state):
    # Pause for human input
    print("Awaiting human approval...")
    print(f"Current state: {state}")
    
    # In a real application, this would integrate with a UI
    approval = input("Approve? (y/n): ")
    
    if approval.lower() == 'y':
        return {"approved": True}
    else:
        return {"approved": False, "needs_revision": True}

# Add human node to workflow
workflow.add_node("human_approval", human_approval_node)
workflow.add_edge("writer", "human_approval")
workflow.add_conditional_edges("human_approval", 
    lambda x: "reviewer" if x.get("approved") else "writer",
    {"reviewer": "reviewer", "writer": "writer"}
)
```

### LangGraph Use Cases

1. **Complex Agent Workflows**: Multi-step reasoning, planning, and execution
2. **Multi-Agent Systems**: Collaborative AI systems with specialized agents
3. **Interactive Applications**: Chatbots with memory and context
4. **Automated Workflows**: Business process automation with AI
5. **Research and Analysis**: Multi-step research and synthesis tasks

---

## Detailed Comparison Analysis

### Feature Comparison Matrix

| Feature | LiteLLM | LangGraph |
|---------|---------|-----------|
| **Primary Purpose** | API Gateway/Proxy | Workflow Framework |
| **Abstraction Level** | Model Provider | Application Logic |
| **State Management** | Stateless | Stateful |
| **Execution Model** | Request/Response | Graph-based |
| **Multi-Provider Support** | ✅ Excellent | ❌ No (uses LangChain) |
| **Cost Tracking** | ✅ Built-in | ❌ No |
| **Fallback/Retry** | ✅ Built-in | ❌ Manual |
| **Streaming** | ✅ Native | ✅ Native |
| **Async Support** | ✅ Yes | ✅ Yes |
| **Memory/Persistence** | ❌ No | ✅ Built-in |
| **Multi-Agent** | ❌ No | ✅ Core Feature |
| **Conditional Logic** | ❌ No | ✅ Core Feature |
| **Human-in-Loop** | ❌ No | ✅ Built-in |
| **Observability** | ✅ Excellent | ✅ Good |

### Strengths and Weaknesses

#### LiteLLM Strengths
- **Universal API**: Works with 100+ LLM providers
- **Zero Code Changes**: Switch providers without modifying application code
- **Production Ready**: Built-in monitoring, cost tracking, and reliability features
- **Performance**: Minimal latency overhead
- **Standardization**: Consistent interface across all providers

#### LiteLLM Weaknesses
- **No State Management**: Cannot handle complex, stateful interactions
- **Limited Logic**: No built-in conditional routing or workflow capabilities
- **Simple Patterns**: Not suitable for complex agent behaviors
- **No Memory**: Cannot maintain context across separate requests

#### LangGraph Strengths
- **Complex Workflows**: Handles multi-step, branching logic
- **State Management**: Maintains context and memory across interactions
- **Agent Patterns**: Built for sophisticated AI agent behaviors
- **Flexibility**: Highly customizable workflow patterns
- **Persistence**: Built-in checkpointing and resume capabilities

#### LangGraph Weaknesses
- **Single Provider**: Tied to LangChain's model abstractions
- **Complexity**: Steeper learning curve for simple use cases
- **Overhead**: More complex than needed for basic LLM calls
- **Dependencies**: Heavy dependency on LangChain ecosystem

---

## Integration Patterns

### Pattern 1: LangGraph + LiteLLM Proxy

```python
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from typing import TypedDict

# Configure LangChain to use LiteLLM proxy
llm = ChatOpenAI(
    model="gpt-4o",
    base_url="http://localhost:4000",  # LiteLLM proxy
    api_key="your-master-key"
)

class IntegratedState(TypedDict):
    query: str
    result: str
    provider_used: str

def llm_node(state: IntegratedState):
    response = llm.invoke([{"role": "user", "content": state["query"]}])
    return {
        "result": response.content,
        "provider_used": "via_litellm_proxy"
    }

# Build integrated workflow
workflow = StateGraph(IntegratedState)
workflow.add_node("llm_call", llm_node)
workflow.set_entry_point("llm_call")
workflow.add_edge("llm_call", END)

app = workflow.compile()
```

### Pattern 2: Custom LLM Integration

```python
from litellm import completion
from langgraph.graph import StateGraph, END
from typing import TypedDict, List

class CustomLLMState(TypedDict):
    messages: List[dict]
    current_provider: str
    fallback_providers: List[str]

def smart_llm_node(state: CustomLLMState):
    providers = [state["current_provider"]] + state["fallback_providers"]
    
    for provider in providers:
        try:
            response = completion(
                model=provider,
                messages=state["messages"],
                timeout=30
            )
            return {
                "messages": state["messages"] + [{"role": "assistant", "content": response.choices[0].message.content}],
                "current_provider": provider
            }
        except Exception as e:
            print(f"Provider {provider} failed: {e}")
            continue
    
    raise Exception("All providers failed")

# Build resilient workflow
workflow = StateGraph(CustomLLMState)
workflow.add_node("smart_llm", smart_llm_node)
workflow.set_entry_point("smart_llm")
workflow.add_edge("smart_llm", END)

app = workflow.compile()

# Usage
inputs = {
    "messages": [{"role": "user", "content": "Hello"}],
    "current_provider": "gpt-4o",
    "fallback_providers": ["claude-3-sonnet-20240229", "gpt-3.5-turbo"]
}

result = app.invoke(inputs)
```

### Pattern 3: Production Architecture

```python
# production_config.py
from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class ProductionConfig:
    litellm_proxy_url: str
    litellm_api_key: str
    default_models: List[str]
    fallback_models: List[str]
    max_retries: int
    timeout: int
    enable_streaming: bool
    enable_checkpointing: bool
    database_url: str

# production_workflow.py
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from litellm import completion
import sqlite3
from typing import TypedDict, List, Optional

class ProductionState(TypedDict):
    session_id: str
    user_query: str
    conversation_history: List[Dict[str, str]]
    current_step: str
    results: Dict[str, Any]
    error_count: int
    last_provider: Optional[str]

class ProductionWorkflow:
    def __init__(self, config: ProductionConfig):
        self.config = config
        self.checkpointer = SqliteSaver(
            sqlite3.connect(config.database_url, check_same_thread=False)
        ) if config.enable_checkpointing else None
        
    def resilient_llm_call(self, state: ProductionState) -> ProductionState:
        models = self.config.default_models + self.config.fallback_models
        
        for model in models:
            try:
                response = completion(
                    model=model,
                    messages=state["conversation_history"],
                    timeout=self.config.timeout,
                    max_retries=self.config.max_retries,
                    api_base=self.config.litellm_proxy_url,
                    api_key=self.config.litellm_api_key
                )
                
                return {
                    **state,
                    "results": {"response": response.choices[0].message.content},
                    "last_provider": model,
                    "error_count": 0
                }
                
            except Exception as e:
                print(f"Model {model} failed: {e}")
                continue
        
        return {
            **state,
            "error_count": state["error_count"] + 1,
            "results": {"error": "All models failed"}
        }
    
    def build_workflow(self) -> StateGraph:
        workflow = StateGraph(ProductionState)
        workflow.add_node("llm_call", self.resilient_llm_call)
        workflow.set_entry_point("llm_call")
        workflow.add_edge("llm_call", END)
        
        return workflow.compile(checkpointer=self.checkpointer)

# Usage
config = ProductionConfig(
    litellm_proxy_url="http://localhost:4000",
    litellm_api_key="your-key",
    default_models=["gpt-4o", "claude-3-sonnet-20240229"],
    fallback_models=["gpt-3.5-turbo"],
    max_retries=3,
    timeout=30,
    enable_streaming=True,
    enable_checkpointing=True,
    database_url="production.db"
)

workflow_manager = ProductionWorkflow(config)
app = workflow_manager.build_workflow()
```

---

## Decision Framework

### When to Use LiteLLM

✅ **Use LiteLLM when you need:**
- **Provider Flexibility**: Ability to switch between different LLM providers
- **Cost Optimization**: Routing requests based on cost and performance
- **Standardization**: Consistent API across different models
- **Reliability**: Built-in fallback and retry mechanisms
- **Monitoring**: Cost tracking and usage analytics
- **Simple Integration**: Drop-in replacement for existing LLM calls

### When to Use LangGraph

✅ **Use LangGraph when you need:**
- **Complex Workflows**: Multi-step reasoning and decision-making
- **State Management**: Maintaining context across interactions
- **Agent Patterns**: ReAct, planning, tool-using agents
- **Conditional Logic**: Branching workflows based on results
- **Multi-Agent Systems**: Collaborative AI agents
- **Human-in-the-Loop**: Interactive approval and feedback
- **Long-running Processes**: Checkpointing and resume capabilities

### When to Use Both Together

✅ **Use Both when you need:**
- **Complex Applications with Provider Flexibility**: LangGraph for workflow logic, LiteLLM for model access
- **Production-Grade Agent Systems**: LangGraph for orchestration, LiteLLM for reliability
- **Multi-Provider Agent Workflows**: Best of both worlds
- **Scalable AI Applications**: LangGraph for logic, LiteLLM for infrastructure

---

## Production Considerations

### Performance Considerations

#### LiteLLM Performance
- **Latency**: Minimal overhead (~5-10ms)
- **Throughput**: Scales with underlying provider limits
- **Memory**: Low memory footprint
- **Caching**: Built-in response caching

#### LangGraph Performance
- **Latency**: Higher due to state management
- **Throughput**: Depends on workflow complexity
- **Memory**: Higher due to state persistence
- **Scaling**: Requires careful state management

### Monitoring and Observability

#### LiteLLM Monitoring
```python
# Built-in observability
litellm.success_callback = ["langfuse", "lunary", "helicone"]
litellm.failure_callback = ["sentry"]

# Custom monitoring
def custom_monitor(kwargs, completion_response, start_time, end_time):
    latency = end_time - start_time
    cost = completion_response.get("cost", 0)
    model = kwargs.get("model")
    
    # Log to your monitoring system
    logger.info(f"LLM Call: {model}, Latency: {latency}s, Cost: ${cost}")

litellm.success_callback = [custom_monitor]
```

#### LangGraph Monitoring
```python
from langgraph.graph import StateGraph
import logging

# Custom node with monitoring
def monitored_node(state):
    start_time = time.time()
    try:
        result = actual_node_logic(state)
        logger.info(f"Node completed in {time.time() - start_time}s")
        return result
    except Exception as e:
        logger.error(f"Node failed: {e}")
        raise

# Add monitoring to workflow
workflow.add_node("monitored_step", monitored_node)
```

### Error Handling and Resilience

#### LiteLLM Error Handling
```python
from litellm import completion
from litellm.exceptions import AuthenticationError, RateLimitError

def robust_completion(messages, models):
    for model in models:
        try:
            return completion(
                model=model,
                messages=messages,
                max_retries=3,
                timeout=30
            )
        except AuthenticationError:
            logger.error(f"Auth failed for {model}")
        except RateLimitError:
            logger.warning(f"Rate limit hit for {model}")
        except Exception as e:
            logger.error(f"Unexpected error with {model}: {e}")
    
    raise Exception("All models failed")
```

#### LangGraph Error Handling
```python
def error_handling_node(state):
    try:
        return successful_operation(state)
    except Exception as e:
        return {
            "error": str(e),
            "retry_count": state.get("retry_count", 0) + 1,
            "needs_retry": state.get("retry_count", 0) < 3
        }

def should_retry(state):
    return "retry" if state.get("needs_retry") else "end"

workflow.add_conditional_edges("error_node", should_retry, {
    "retry": "main_logic",
    "end": END
})
```

### Cost Management

#### LiteLLM Cost Optimization
```python
# Cost-aware routing
def cost_aware_completion(messages, budget_limit=1.0):
    models_by_cost = [
        ("gpt-3.5-turbo", 0.001),
        ("gpt-4o-mini", 0.01),
        ("gpt-4o", 0.1)
    ]
    
    for model, estimated_cost in models_by_cost:
        if estimated_cost <= budget_limit:
            try:
                return completion(model=model, messages=messages)
            except:
                continue
    
    raise Exception("No model within budget")
```

## Key Takeaways

This comprehensive guide provides the foundation for understanding and implementing both LiteLLM and LangGraph in production environments. The key insights are:

1. **LiteLLM excels as an infrastructure layer** - providing unified access, cost management, and reliability across multiple LLM providers
2. **LangGraph excels as an application layer** - enabling complex workflows, state management, and agent patterns
3. **They complement each other perfectly** - LiteLLM handles the "how" of LLM access, while LangGraph handles the "what" of workflow logic
4. **Production success requires both** - LiteLLM for robustness and provider flexibility, LangGraph for sophisticated AI behaviors

Choose LiteLLM when you need provider abstraction and infrastructure reliability. Choose LangGraph when you need complex workflows and stateful interactions. Use both together for production-grade AI applications that are both robust and sophisticated.