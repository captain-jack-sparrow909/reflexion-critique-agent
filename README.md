### How reflexion agent work theoritically ?
- user send a request, first responder node which will respond and along with initial response it'll also add 2 additional things 1. critique 2. search
 first-responder ->
    1. response :
    2. critique :  
    3. search : search terms that would be beneficials to get better response

- second comes the execute tools node, which will simply take our search queries and use search engine to give us result in real time

- third comes the revisor node, which will take the initial response + results of search engine execution, it'll revisit and change the original response
while taking the initial critique in mind, also it'll generate a new critique, search terms to the revised response, and it'll include citation of the first search that we had

- 4th we are in loop of tool execution (searching) and revision until we come to a stopping position and then a response is sent back to the user.










### reflexion agent diagram: image.png





# when to use: llm.with_structured_output and llm.bind_tools(tools=[ClassName], tool_choice="ClassName"):
```
chain = prompt | llm.with_structured_output(AnswerQuestion)

result = chain.invoke(...)
print(type(result))   # <class 'AnswerQuestion'>  ← Python object directly
print(result.answer)  # "Paris"                  ← access fields directly
```
LangChain handles the tool call internally and returns a clean Pydantic object. The raw tool call message never surfaces.

while:
```
chain = prompt | llm.bind_tools(tools=[AnswerQuestion], tool_choice="AnswerQuestion")

result = chain.invoke(...)
print(type(result))   # <class 'AIMessage'>   ← still a message object
print(result.tool_calls)  # [{"name": "AnswerQuestion", "args": {"answer": "Paris", ...}}]
```
The raw AIMessage with the tool_calls field is returned. You are responsible for parsing it.


With with_structured_output, the raw AIMessage is swallowed — you get a plain Pydantic object back, which can't be stored in message history in a meaningful way for other nodes to reason about.

**why**:

## It's Not About `with_structured_output` Itself

The response not being stored has nothing to do with `with_structured_output` directly. It's about **what type of object gets returned** and whether it fits in the messages list.

---

## The Messages List Expects Message Objects

```python
class MyState(TypedDict):
    messages: Annotated[list, add_messages]
```

`add_messages` reducer expects objects like:
- `AIMessage`
- `HumanMessage`
- `ToolMessage`

---

## What Each Approach Returns

```python
# bind_tools → returns AIMessage ✅ fits in messages list
result = chain_with_bind_tools.invoke(...)
print(type(result))  # AIMessage  ← can be stored in state

# with_structured_output → returns Pydantic object ❌ doesn't fit
result = chain_with_structured_output.invoke(...)
print(type(result))  # AnswerQuestion  ← not a message object
```

---

## So Practically Speaking

```python
# With bind_tools — works naturally with state
def generation_node(state):
    return {"messages": [chain.invoke(...)]}  # AIMessage goes in ✅

# With with_structured_output — you'd have to manually wrap it
def generation_node(state):
    result = chain.invoke(...)   # AnswerQuestion object
    # result is a Pydantic object, not a message
    # you CAN'T just do {"messages": [result]}
    # you'd have to do something like:
    return {"messages": [AIMessage(content=result.json())]}  # manual wrapping ❌ awkward
```

---

## TL;DR

`with_structured_output` **can** technically be stored if you manually wrap it, but it defeats the purpose. In LangGraph, `bind_tools` is the natural fit because the `AIMessage` it returns **slots directly into message history** without any extra work, and downstream nodes can inspect the tool call natively.





--------------
execute_tools = ToolNode(
    [
        StructuredTool.from_function(run_queries, name=AnswerQuestion.__name__),
        StructuredTool.from_function(run_queries, name=ReviseAnswer.__name__)
    ]
)

When You Do bind_tools(tools=[AnswerQuestion])
The LLM receives AnswerQuestion as a tool definition with its name and schema:
{
  "name": "AnswerQuestion",        ← the Pydantic class name becomes the tool name
  "description": "...",
  "parameters": {
    "answer": {"type": "string"},
    "reflection": {"type": "string"},
    "search_queries": {"type": "array"}
  }
}

The LLM Thinks It's "Calling a Tool"
From the LLM's perspective, it doesn't know AnswerQuestion is just a Pydantic schema. It sees it as a callable tool it must invoke. So it produces:
{
  "name": "AnswerQuestion",       ← echoes back the tool name it was given
  "args": {
    "answer": "Paris is the capital...",
    "reflection": "The question was straightforward",
    "search_queries": ["capital of France", "Paris history"]
  }
}


The Problem
AnswerQuestion and ReviseAnswer are Pydantic schemas, not real executable tools:
They were only used to force structured output from the LLM. But when ToolNode receives a tool call named "AnswerQuestion", it needs an actual executable function to run — not just a schema.
What ToolNode Does With It
ToolNode receives:
{"name": "AnswerQuestion", "args": {"answer": "...", "search_queries": ["q1", "q2"]}}
        ↓
looks for a registered tool named "AnswerQuestion"  ← finds StructuredTool
        ↓
calls run_queries(search_queries=["q1", "q2"])      ← **kwargs absorbs "answer", "reflection"
        ↓
runs actual Tavily searches and returns results
