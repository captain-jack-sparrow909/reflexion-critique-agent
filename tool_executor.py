from langchain_core.tools import StructuredTool
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv
from langchain_tavily import TavilySearch

from schemas import AnswerQuestion, ReviseAnswer


load_dotenv()

# StructuredTool.from_function() is just the explicit/manual version of @tool.
# LangChain's basic Tool only accepts a single string input:
# # Basic Tool — single string input only
# tool = Tool(name="search", func=lambda query: search(query))
# But real functions often need multiple arguments, this is where StructuredTool can help you
# but @tool does the same thing, so if you can use @tool
# from langchain_core.tools import StructuredTool

# def multiply(a: int, b: int) -> int:
#     """Multiply two numbers."""
#     return a * b

# tool = StructuredTool.from_function(multiply)
# for below scenario we can't use, becuase we're trying to register same function with different names

tavily_tool = TavilySearch(max_results=3)


def run_queries(search_queries: list[str], **kwargs):
    """Run the generated queries"""
    return tavily_tool.batch([{"query": query} for query in search_queries])


execute_tools = ToolNode(
    [
        StructuredTool.from_function(run_queries, name=AnswerQuestion.__name__),
        StructuredTool.from_function(run_queries, name=ReviseAnswer.__name__)
    ]
)

# When You Do bind_tools(tools=[AnswerQuestion])
# The LLM receives AnswerQuestion as a tool definition with its name and schema:
# {
#   "name": "AnswerQuestion",        ← the Pydantic class name becomes the tool name
#   "description": "...",
#   "parameters": {
#     "answer": {"type": "string"},
#     "reflection": {"type": "string"},
#     "search_queries": {"type": "array"}
#   }
# }

# The LLM Thinks It's "Calling a Tool"
# From the LLM's perspective, it doesn't know AnswerQuestion is just a Pydantic schema. It sees it as a callable tool it must invoke. So it produces:
# {
#   "name": "AnswerQuestion",       ← echoes back the tool name it was given
#   "args": {
#     "answer": "Paris is the capital...",
#     "reflection": "The question was straightforward",
#     "search_queries": ["capital of France", "Paris history"]
#   }
# }


# The Problem
# AnswerQuestion and ReviseAnswer are Pydantic schemas, not real executable tools:
# They were only used to force structured output from the LLM. But when ToolNode receives a tool call named "AnswerQuestion", it needs an actual executable function to run — not just a schema.
# What ToolNode Does With It
# ToolNode receives:
# {"name": "AnswerQuestion", "args": {"answer": "...", "search_queries": ["q1", "q2"]}}
#         ↓
# looks for a registered tool named "AnswerQuestion"  ← finds StructuredTool
#         ↓
# calls run_queries(search_queries=["q1", "q2"])      ← **kwargs absorbs "answer", "reflection"
#         ↓
# runs actual Tavily searches and returns results
