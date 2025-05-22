import docker
import time
import os
from datetime import date, timedelta, datetime
from duckduckgo_search import DDGS
from fredapi import Fred
from functools import lru_cache
from tavily import TavilyClient
from langchain_community.utilities import SerpAPIWrapper
from langchain_community.tools import BraveSearch
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper
from langchain.tools import Tool
from langchain_community.tools.yahoo_finance_news import YahooFinanceNewsTool
from typing import List, Dict, Any, Optional, Union
import yfinance as yf
import pandas as pd

import config

YESTERDAY = (date.today() - timedelta(days=1)).strftime('%Y-%m-%d')

# --- Docker Utilities ---
docker_client = None
try:
    docker_client = docker.from_env()
except docker.errors.DockerException:
    print("TOOL_SHED: Docker is not running or accessible. Code Agent will not function.")


def build_docker_image_once():
    """Builds the Docker image for the Code Agent if it doesn't exist."""
    if not docker_client:
        return False
    try:
        docker_client.images.get(config.DOCKER_IMAGE_NAME)
        print(f"TOOL_SHED: Docker image '{config.DOCKER_IMAGE_NAME}' found.")
        return True
    except docker.errors.ImageNotFound:
        print(
            f"TOOL_SHED: Docker image '{config.DOCKER_IMAGE_NAME}' not found. "
            "Building..."
        )
        try:
            # Assuming Dockerfile is in the same directory as main.py
            dockerfile_path = os.path.abspath(
                os.path.join(os.path.dirname(__file__), '.')
            )  # project root
            docker_client.images.build(
                path=dockerfile_path, 
                tag=config.DOCKER_IMAGE_NAME,
                rm=True
            )
            print(
                f"TOOL_SHED: Docker image '{config.DOCKER_IMAGE_NAME}' "
                "built successfully."
            )
            return True
        except docker.errors.BuildError as e:
            print(f"TOOL_SHED: Failed to build Docker image: {e}")
            for line in e.build_log:
                if 'stream' in line:
                    print(line['stream'].strip())
            return False
    except Exception as e:
        print(f"TOOL_SHED: Error checking/building Docker image: {e}")
        return False


# Call this once at application startup
IS_DOCKER_IMAGE_READY = build_docker_image_once()


def install_libraries(libraries: List[str]) -> Dict[str, Any]:
    """Installs libraries in the Code Agent's Docker image (commits a new layer).

    Args:
        libraries (List[str]): A list of library names to install.

    Returns:
        Dict[str, Any]: A dictionary containing the status and log of the library installation.
    """
    if not IS_DOCKER_IMAGE_READY or not docker_client:
        return {
            "status": "failure", 
            "log": "Docker image not ready or Docker not running."
        }
    if not libraries:
        return {
            "status": "success",
            "log": "No libraries requested for installation."
        }

    try:
        container = docker_client.containers.run(
            config.DOCKER_IMAGE_NAME,
            command=["pip", "install"] + libraries,
            detach=False,  # Run and wait
            remove=False   # We need to commit this container
        )
        logs = container.logs().decode('utf-8')

        # Check if installation was successful
        if ("Successfully installed" in logs or 
                all(f"Requirement already satisfied: {lib}" in logs 
                    for lib in libraries)):
            print(f"TOOL_SHED: Libraries installation log:\n{logs}")
            container.remove()  # Clean up container
            return {"status": "success", "log": logs}
        else:
            print(f"TOOL_SHED: Library installation failed:\n{logs}")
            container.remove()
            return {"status": "failure", "log": logs}

    except docker.errors.ContainerError as e:
        log_output = (e.stderr.decode('utf-8') if e.stderr
                      else "Unknown container error")
        print(f"TOOL_SHED: ContainerError: {log_output}")
        if hasattr(e, 'container') and e.container:
            e.container.remove()
        return {"status": "failure", "log": log_output}
    except Exception as e:
        print(f"TOOL_SHED: Exception during library installation: {e}")
        return {"status": "failure", "log": str(e)}


def execute_python_code(code: str, libraries_needed: Optional[List[str]] = None) -> Dict[str, Any]:
    """Executes Python code in an isolated Docker container.

    Args:
        code (str): The Python code to execute.
        libraries_needed (Optional[List[str]]): A list of libraries to install before running the code.

    Returns:
        Dict[str, Any]: A dictionary containing the status, stdout, and stderr of the code execution.
    """
    if not IS_DOCKER_IMAGE_READY or not docker_client:
        return {"status": "failure", "stdout": "", "stderr": "Docker image not ready or Docker not running."}

    # Prepare the code: if libraries are needed, prepend pip install commands
    final_code_to_run = ""
    if libraries_needed:
        install_commands = "\n".join([f"import subprocess; subprocess.run(['pip', 'install', '{lib}'], check=True)" for lib in libraries_needed])
        final_code_to_run = f"import sys\n{install_commands}\n\n# User code starts here\n{code}"
    else:
        final_code_to_run = code

    # Create a temporary script file to mount into the container
    temp_script_name = f"temp_script_{int(time.time())}.py"
    with open(temp_script_name, "w") as f:
        f.write(final_code_to_run)

    abs_script_path = os.path.abspath(temp_script_name)
    script_dir = os.path.dirname(abs_script_path)
    container_script_path = f"/app/{temp_script_name}"

    container = None
    try:
        # Run the container with the script mounted
        container = docker_client.containers.run(
            config.DOCKER_IMAGE_NAME,
            command=["python", container_script_path],
            volumes={script_dir: {'bind': '/app', 'mode': 'rw'}},
            working_dir="/app",
            stderr=True,
            stdout=True,
            detach=True,  # Run in background
            remove=False  # Don't auto-remove so we can get logs
        )

        # Wait for container to finish with timeout
        exit_code = container.wait(timeout=config.DOCKER_CONTAINER_TIMEOUT)["StatusCode"]

        # Get logs
        stdout_log = container.logs(stdout=True, stderr=False).decode('utf-8')
        stderr_log = container.logs(stdout=False, stderr=True).decode('utf-8')

        # Clean up
        try:
            container.remove()
        except Exception as e:
            print(f"TOOL_SHED: Warning - Could not remove container: {e}")

        if os.path.exists(abs_script_path):
            os.remove(abs_script_path)

        if exit_code == 0:
            return {"status": "success", "stdout": stdout_log, "stderr": stderr_log}
        else:
            return {"status": "failure", "stdout": stdout_log, "stderr": stderr_log}

    except docker.errors.ContainerError as e:
        # This typically means the Python script had an error (exit code > 0)
        stdout_log = ""
        stderr_log = str(e)

        if container:
            try:
                stdout_log = container.logs(stdout=True, stderr=False).decode('utf-8')
                stderr_log = container.logs(stdout=False, stderr=True).decode('utf-8')
                container.remove()
            except Exception as container_e:
                print(f"TOOL_SHED: Error getting logs or removing container: {container_e}")

        if os.path.exists(abs_script_path):
            os.remove(abs_script_path)

        return {"status": "failure", "stdout": stdout_log, "stderr": stderr_log}

    except docker.errors.NotFound as e:
        # Container not found - could happen if Docker removed it
        print(f"TOOL_SHED: Container not found error: {e}")
        if os.path.exists(abs_script_path):
            os.remove(abs_script_path)
        return {"status": "failure", "stdout": "", "stderr": f"Docker container error: {str(e)}"}

    except Exception as e:
        print(f"TOOL_SHED: Exception during code execution: {e}")
        if container:
            try:
                container.remove()
            except Exception:
                pass  # Already gone or can't be removed

        if os.path.exists(abs_script_path):
            os.remove(abs_script_path)

        return {"status": "failure", "stdout": "", "stderr": str(e)}


def search_duckduckgo(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """Searches DuckDuckGo for the given query and returns a list of results.

    Args:
        query (str): The search query.
        max_results (int): The maximum number of results to return.

    Returns:
        List[Dict[str, str]]: A list of search results.
    """
    print(f"TOOL_SHED: Searching DuckDuckGo for: '{query}'")
    try:
        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=max_results)
            return [{"title": r.get('title'), "body": r.get('body'),
                     "href": r.get('href'), 
                     "source": "DuckDuckGo"} for r in results] if results else []
    except Exception as e:
        print(f"TOOL_SHED: DuckDuckGo search error: {e}")
        return [{"error": str(e), "source": "DuckDuckGo"}]


def search_tavily(query: str, search_depth: str = "basic", max_results: int = 3) -> List[Dict[str, Any]]:
    """Searches Tavily for the given query and returns a list of results.

    Args:
        query (str): The search query.
        search_depth (str): The depth of the search.
        max_results (int): The maximum number of results to return.

    Returns:
        List[Dict[str, Any]]: A list of search results.
    """
    print(f"TOOL_SHED: Searching Tavily for: '{query}'")
    if not config.TAVILY_API_KEY:
        print("TOOL_SHED: Tavily API key not found.")
        return [{"error": "Tavily API key not configured", "source": "Tavily"}]
    try:
        client = TavilyClient(api_key=config.TAVILY_API_KEY)
        response = client.search(query=query, search_depth=search_depth,
                                 max_results=max_results, include_answer=True)
        # response structure: {'query': '...', 'follow_up_questions': ..., 'answer': '...', 'images': ..., 'results': [...], 'response_time': ...}
        tavily_results = []
        if response.get("answer"):
            tavily_results.append({"answer": response.get("answer"), "source": "Tavily Answer"})
        if response.get("results"):
            for r in response.get("results", []):
                tavily_results.append({"title": r.get('title'),
                                       "content": r.get('content'),
                                       "url": r.get('url'),
                                       "score": r.get("score"),
                                       "source": "Tavily Search Result"})
        return tavily_results
    except Exception as e:
        print(f"TOOL_SHED: Tavily search error: {e}")
        return [{"error": str(e), "source": "Tavily"}]


def search_wikipedia(query: str) -> List[Dict[str, Any]]:
    """Searches Wikipedia for the given query."""
    print(f"TOOL_SHED: Searching Wikipedia for: '{query}'")
    try:
        wikipedia = WikipediaAPIWrapper()
        results = wikipedia.run(query)
        return [{"content": results, "source": "Wikipedia"}]
    except Exception as e:
        print(f"TOOL_SHED: Wikipedia search error: {e}")
        return [{"error": str(e), "source": "Wikipedia"}]


def search_brave(query: str) -> List[Dict[str, Any]]:
    """Searches Brave for the given query and returns a list of results.

    Args:
        query (str): The search query.

    Returns:
        List[Dict[str, Any]]: A list of search results.
    """
    print(f"TOOL_SHED: MOCK Searching Brave for: '{query}'")
    if not config.BRAVE_SEARCH_API_KEY:
        print("TOOL_SHED: Brave Search API key not found.")
        return [{"error": "Brave Search API key not configured", "source": "BraveSearch"}]
    try:
        brave_search_tool = BraveSearch(api_key=config.BRAVE_SEARCH_API_KEY, search_kwargs={"count": 3})
        results = brave_search_tool.run(query)
        return results
    except Exception as e:
        print(f"TOOL_SHED: Brave search error: {e}")
        return [{"error": str(e), "source": "BraveSearch"}]


def search_serpapi(query: str) -> List[Dict[str, Any]]:
    """Searches SerpAPI for the given query and returns a list of results.

    Args:
        query (str): The search query.

    Returns:
        List[Dict[str, Any]]: A list of search results.
    """
    print(f"TOOL_SHED: Searching SerpAPI for: '{query}'")
    if not config.SERPAPI_API_KEY:
        print("TOOL_SHED: SerpAPI key not found.")
        return [{"error": "SerpAPI key not configured", "source": "SerpAPI"}]
    try:
        serpapi_tool = SerpAPIWrapper(api_key=config.SERPAPI_API_KEY)
        results = serpapi_tool.run(query)
        return results
    except Exception as e:
        print(f"TOOL_SHED: SerpAPI search error: {e}")
        return [{"error": str(e), "source": "SerpAPI"}]


# Mapping tool names to functions
SEARCH_FUNCTION_MAP = {
    "DuckDuckGo": search_duckduckgo,
    "Tavily": search_tavily,
    "BraveSearch": search_brave,
    "SerpAPI": search_serpapi,
    "Wikipedia": search_wikipedia,
}


# Financial Data Retrieving Agent specific tools
def get_yahoo_finance_news(ticker_symbol: str) -> List[Dict[str, Any]]:
    """Gets news from Yahoo Finance for the given ticker symbol (i.e. an 
    abbreviation used to uniquely identify publicly traded shares of a 
    particular stock or security on a particular stock exchange)."""
    print(f"TOOL_SHED: Getting Yahoo Finance news for: '{ticker_symbol}'")
    try:
        news_tool = YahooFinanceNewsTool(ticker_symbol)
        results = news_tool.invoke()
        return results
    except Exception as e:
        print(f"TOOL_SHED: Yahoo Finance News error: {e}")
        return [{"error": str(e), "source": "Yahoo Finance News"}]


@lru_cache(maxsize=16)
def get_fred_index_data(economical_index: str,
                        start_date: str = '1991-01-01',
                        end_date: str = YESTERDAY) -> List[Dict[str, Any]]:
    """Gets data from the Federal Reserve Bank of St. Louis (FRED) for the given 
    economical index.
    Args:
        economical_index (str): The FRED series ID for the economical index.
        start_date (str): The start date for data retrieval in the format YYYY-MM-DD.
        end_date (str): The end date for data retrieval in the format YYYY-MM-DD.

    Returns:
        List[Dict[str, Any]]: A list of FRED data.
    """
    print(f"TOOL_SHED: Getting FRED data for: '{economical_index}'")
    if not config.FRED_API_KEY:
        print("TOOL_SHED: FRED API key not found.")
        return [{"error": "FRED API key not configured", "source": "FRED"}]
    try:
        fred = Fred(api_key=config.FRED_API_KEY)
        results = fred.get_series_latest_release(economical_index,
                                                 start_date=start_date,
                                                 end_date=end_date)
        return results.to_dict()
    except Exception as e:
        print(f"TOOL_SHED: FRED data retrieval error: {e}")
        return [{"error": str(e), "source": "FRED"}]


def get_historical_yahoo_finance_stock_data(ticker_symbol: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> Dict[str, Any]:
    """Retrieves historical stock data from Yahoo Finance."""
    print(f"TOOL_SHED: Getting historical Yahoo Finance stock data for: '{ticker_symbol}'")
    
    # Default to last 7 days if no dates provided
    if not start_date:
        start_date = (date.today() - timedelta(days=7)).strftime('%Y-%m-%d')
    if not end_date:
        end_date = date.today().strftime('%Y-%m-%d')
    
    # Parse dates to ensure they're in the correct format
    try:
        parsed_start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
        parsed_end_date = datetime.strptime(end_date, '%Y-%m-%d').date()
    except ValueError:
        return [{"error": "Invalid date format. Use YYYY-MM-DD.", "source": "Yahoo Finance"}]
    
    if parsed_start_date >= parsed_end_date:
        return [{"error": "Start date must be before end date", "source": "Yahoo Finance"}]

    try:
        ticker = yf.Ticker(ticker_symbol)
        historical_data = ticker.history(
            start=parsed_start_date,
            end=parsed_end_date,
            interval='1d'
        )
        
        # Convert to dictionary format
        result = {}
        for column in historical_data.columns:
            result[column] = historical_data[column].to_dict()
            
        return result

    except Exception as e:
        print(f"TOOL_SHED: Error fetching data for {ticker_symbol}: {e}")
        return [{"error": str(e), "source": "Yahoo Finance"}]


FINANCIAL_DATA_RETRIEVAL_TOOLS = {
    "Yahoo Finance": get_historical_yahoo_finance_stock_data,
    "Yahoo Finance News": get_yahoo_finance_news,
    "FRED": get_fred_index_data
}
