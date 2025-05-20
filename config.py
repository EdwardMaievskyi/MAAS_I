import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
BRAVE_SEARCH_API_KEY = os.getenv("BRAVE_SEARCH_API_KEY")

# Search tool priority
SEARCH_TOOL_PRIORITY = ["DuckDuckGo", "BraveSearch", "Tavily", "SerpAPI"]

# Docker configuration for Code Agent
DOCKER_IMAGE_NAME = "maas_code_agent_env:latest"
DOCKER_CONTAINER_TIMEOUT = 300

# LLM Configuration
ORCHESTRATOR_LLM_MODEL = "gpt-4.1-mini-2025-04-14"
CODE_GENERATION_LLM_MODEL = "o4-mini-2025-04-16"
LLM_TEMPERATURE = 0.7
