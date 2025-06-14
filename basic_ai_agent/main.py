from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import asyncio
import os

load_dotenv()

openai_api_key = os.getenv('OPENAI_API_KEY')
firecrawl_api_key = os.getenv('FIRECRAWL_API_KEY')