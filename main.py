from typing import Literal
from langchain_core.messages import ToolMessage
from langgraph.graph import END, START, StateGraph, MessagesState
from tool_executor import execute_tools
from chains import revisor_chain, first_responder_chain


# nodes helper functions:
def draft_node(state: MessagesState):
    """draft the initial response"""
    response = first_responder_chain.invoke({"messages": state["messages"]})  # MessagesPlaceholder dynamically injects the entire conversation history into the prompt. So passing state["messages"] gives the LLM full context of everything said so far — the original question, previous drafts, reflections etc.
    return {"messages": [response]}  # There is no PydanticToolsParser at the end of first responder chain. So the chain returns a raw AIMessage object, not a dict or Pydantic object:
    #what is returned It goes directly into the LangGraph state via the add_messages reducer, since we're using MessagesState


def revise_node(state: MessagesState):
    """Revise the answer based on tool results."""
    response = revisor_chain.invoke({"messages": state["messages"]})
    return {"messages": [response]}


def event_loop(state: MessagesState)->Literal["execute_tools", END]:
    """Determine whether to continue or end based on iteration"""
    count_tool_calls = sum(isinstance(item, ToolMessage) for item in state['messages'])
    if count_tool_calls > 2:
        return END
    return "execute_tools"


# graph:
graph = StateGraph(MessagesState)
graph.add_node("draft", draft_node)
graph.add_node("revisor", revise_node)
graph.add_node("execute_tools", execute_tools)

graph.add_edge(START, "draft")
graph.add_edge("draft", "execute_tools")
graph.add_edge("execute_tools", "revisor")

graph.add_conditional_edges("revisor", event_loop, {
    END: END,
    "execute_tools": "execute_tools"
})

app = graph.compile()


def main():
    print("Hello from reflexion-critique-agent!")
    res = app.invoke({
        "messages": [
            {
                "role": "user",
                "content": "Write about AI-Powered SOC / autonomous soc problem domain, list startups that do that and raised capital."
            }
        ]
    })
    last_message = res["messages"][-1]
    print("---response---", last_message)


if __name__ == "__main__":
    main()
