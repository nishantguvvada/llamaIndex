
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.workflow import Context, JsonSerializer
import requests
import os
from dotenv import load_dotenv
import asyncio
import json

load_dotenv()

llm = GoogleGenAI(
    model="models/gemini-1.5-flash",
    api_key=f"{os.getenv("GEMINI_API_KEY")}",
)

# response = llm.complete("Describe Eartho in two words")
# print(response)

# When creating a tool, its very important to:
# give the tool a proper name and docstring/description. The LLM uses this to understand what the tool does.
# annotate the types. This helps the LLM understand the expected input and output types.
# use async when possible, since this will make the workflow more efficient.

async def iss_position():
    """Useful to fetch International Space Station's exact current position over the Earth"""
    response = requests.get(url="http://api.open-notify.org/iss-now.json")
    return response.json()

agent = FunctionAgent(
    tools=[iss_position],
    llm=llm,
    system_prompt="You are subject matter expert on International Space Station that can fetch current position from live tracking API",
)

# State is stored in the Context. This can be passed between runs to maintain state and history.
ctx = Context(agent)

async def main():

    # Fetching previously stored context
    with open("context.json", "r") as context:
        ctx_dict = json.load(context)

    # Restoring the context from the fetched context dictionary
    restored_ctx = Context.from_dict(agent, ctx_dict, serializer=JsonSerializer())

    # response = await agent.run(user_msg="Hi, my name is Nishant. I want to know where is the International Space Station right now?", ctx=restored_ctx)
    # print(str(response))
    response = await agent.run(user_msg="What question did I ask regarding ISS before?", ctx=restored_ctx)
    print(str(response))

    # Converting the context to the dictionary
    ctx_dict = restored_ctx.to_dict(serializer=JsonSerializer())

    # Storing the context
    with open("context.json", "w") as context:
        json.dump(ctx_dict, context)


asyncio.run(main())

# Use the below code to create a workflow with multiple agents

# from llama_index.core.agent.workflow import AgentWorkflow

# workflow = AgentWorkflow(agents=[agent])

# response = await workflow.run(user_msg="What is the weather in San Francisco?")