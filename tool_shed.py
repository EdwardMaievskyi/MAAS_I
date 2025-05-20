import docker
import time
import os
from duckduckgo_search import DDGS
from tavily import TavilyClient
from langchain_community.utilities import SerpAPIWrapper
from langchain_community.tools import BraveSearch
from typing import List, Dict, Any, Optional

import config

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
    """Executes Python code in an isolated Docker container."""
    if not IS_DOCKER_IMAGE_READY or not docker_client:
        return {"status": "failure", "stdout": "", "stderr": "Docker image not ready or Docker not running."}

    # Prepare the code: if libraries are needed, prepend pip install commands
    # This ensures libraries are installed fresh for each execution, providing better isolation.
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

    try:
        container = docker_client.containers.run(
            config.DOCKER_IMAGE_NAME,
            command=["python", container_script_path],
            volumes={script_dir: {'bind': '/app', 'mode': 'rw'}}, # Mount the directory containing the script
            working_dir="/app",
            stderr=True,
            stdout=True,
            detach=False,  # Run and wait for completion
            remove=True,  # Remove container after execution
            # Consider network_disabled=True for extra security if code doesn't need internet
        )
        # Logs are combined if detach=False and container is removed.
        # For separate stdout/stderr, you might need to stream logs or inspect container before removal.
        # However, for simplicity, `container.logs()` gives combined output.
        # If using `detach=True` and `container.wait()`:
        # stdout = container.logs(stdout=True, stderr=False).decode('utf-8')
        # stderr = container.logs(stdout=False, stderr=True).decode('utf-8')

        # For `detach=False` (synchronous run), the result of `run` is the logs if `remove=True` immediately
        # Let's get logs before it's auto-removed, or trust the output from the run command.
        # The `container` object here *is* the logs if `remove=True` and not detached.
        output_bytes = container  # This is the logs if remove=True
        output_str = output_bytes.decode('utf-8')

        # A simple way to differentiate stdout/stderr is to have the script print markers,
        # or structure its output (e.g., JSON with 'stdout' and 'stderr' keys).
        # For now, we treat all output as potential stdout and will rely on script exit code (implicitly via ContainerError).

        # This is a simplification. `docker.errors.ContainerError` will be raised if script exits non-zero.
        # If no error, all output is considered stdout.
        os.remove(abs_script_path)  # Clean up temp file
        return {"status": "success", "stdout": output_str, "stderr": ""}

    except docker.errors.ContainerError as e:
        # This typically means the Python script had an error (exit code > 0)
        stdout_log = e.container.logs(stdout=True, stderr=False).decode('utf-8')
        stderr_log = e.container.logs(stdout=False, stderr=True).decode('utf-8')
        print(f"TOOL_SHED: ContainerError during code execution. Stdout: {stdout_log}, Stderr: {stderr_log}")
        if os.path.exists(abs_script_path): os.remove(abs_script_path)
        return {"status": "failure", "stdout": stdout_log, "stderr": stderr_log}
    except Exception as e:
        print(f"TOOL_SHED: Exception during code execution: {e}")
        if os.path.exists(abs_script_path): os.remove(abs_script_path)
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
}
