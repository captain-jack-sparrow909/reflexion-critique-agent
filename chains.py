from datetime import datetime
from dotenv import load_dotenv
from langchain_core.output_parsers.openai_tools import JsonOutputToolsParser, PydanticToolsParser
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

from schemas import AnswerQuestion

load_dotenv()

llm = ChatOpenAI(model="gpt-5-nano")

first_responder_prompt = ChatPromptTemplate.from_messages([

           ( "system",
            """You are expert researcher.
                Current time: {time}

                1. {first_instruction}
                2. Reflect and critique your answer. Be severe to maximize improvement.
                3. Recommend search queries to research information and improve your answer.""",
            ),
        MessagesPlaceholder(variable_name="messages")
]).partial(
    time=lambda: datetime.now().isoformat()
)

parser_pydantic = PydanticToolsParser(tools=[AnswerQuestion])

first_responder_prompt_template = first_responder_prompt.partial(
    first_instruction="Provide a detailed ~250 words answer"
)

first_responder_chain = first_responder_prompt_template | llm.bind_tools(tools=[AnswerQuestion], tool_choice="AnswerQuestion") | parser_pydantic


# chain = prompt | llm.with_structured_output(AnswerQuestion)

# result = chain.invoke(...)
# print(type(result))   # <class 'AnswerQuestion'>  ← Python object directly
# print(result.answer)  # "Paris"                  ← access fields directly
# LangChain handles the tool call internally and returns a clean Pydantic object. The raw tool call message never surfaces.


# What with_structured_output Actually Does Internally
# Here's the thing — with_structured_output is essentially a wrapper around bind_tools:

# with_structured_output(AnswerQuestion)
#         ↓ internally does
# bind_tools([AnswerQuestion], tool_choice="AnswerQuestion")
#         +
# JsonOutputToolsParser()   ← this extra step parses the result for you

# note: 
# With with_structured_output, the raw AIMessage is swallowed 
# — you get a plain Pydantic object back, which can't be stored in message history in a meaningful way for other nodes to reason about.



revise_instructions = """Revise your previous answer using the new information.
    - You should use the previous critique to add important information to your answer.
        - You MUST include numerical citations in your revised answer to ensure it can be verified.
        - Add a "References" section to the bottom of your answer (which does not count towards the word limit). In form of:
            - [1] https://example.com
            - [2] https://example.com
    - You should use the previous critique to remove superfluous information from your answer and make SURE it is not more than 250 words.
"""

# this will get plugged into first_intruction

