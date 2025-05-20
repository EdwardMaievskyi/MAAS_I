import json
import uuid
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

import config
from state_model import AgentState, Task
from tool_shed import SEARCH_FUNCTION_MAP, execute_python_code, install_libraries

# Initialize LLMs
orchestrator_llm = ChatOpenAI(model=config.ORCHESTRATOR_LLM_MODEL,
                              temperature=config.LLM_TEMPERATURE, 
                              api_key=config.OPENAI_API_KEY)
code_gen_llm = ChatOpenAI(model=config.CODE_GENERATION_LLM_MODEL,
                          api_key=config.OPENAI_API_KEY)


def create_task_id():
    return str(uuid.uuid4())


def orchestrator_planner_node(state: AgentState) -> AgentState:
    print("AGENT: Orchestrator - Planning...")
    state.overall_status = "planning"
    user_request = state.original_request
    # History of previous attempts if this is a re-plan
    previous_steps_summary = (
        "\nPrevious Execution Log:\n" + 
        "\n".join([
            f"- Task {t.id} ({t.agent_name} - {t.action}): {t.status}, Error: {t.error}" 
            for t in state.executed_tasks_log if t.error
        ]) if state.executed_tasks_log else "No previous attempts."
    )

    prompt = f"""You are an expert planner for a multi-agent system. Your goal is to achieve the user's request by creating a sequence of tasks for other agents.
Available agents and their actions:
1.  SearchAgent:
    -   action: "perform_search"
        details: {{"query": "your search query here"}}
        (Use this to find information on the web.)
2.  CodeAgent:
    -   action: "install_libraries"
        details: {{"libraries": ["lib1", "lib2"]}}
        (Use this *before* executing code if non-standard libraries are needed. The code execution step itself will handle the installation based on this prior step's success.)
    -   action: "generate_code"
        details: {{"prompt": "description of python code to generate", "context_data": "any relevant data from previous steps, like search results"}}
        (Generates Python code. The code should be self-contained or use libraries declared via 'install_libraries'.)
    -   action: "execute_code"
        details: {{"code": "python code string OR 'use_generated_code' if generated in a previous step", "relevant_libraries": ["lib1", "lib2"] (optional, if specific libs from an install step are crucial for this execution run)}}
        (Executes Python code. If "code" is "use_generated_code", it uses the output from the last "generate_code" action. The 'relevant_libraries' will be passed to the execution environment to ensure they are installed before running the code.)

User Request: "{user_request}"
{previous_steps_summary}

Based on the request and any previous errors, create a JSON list of tasks. Each task should have "id", "agent_name", "action", and "details".
Ensure IDs are unique for new tasks.
If the user request is a simple question that can be answered directly, you can create a plan with a single task for yourself:
    {{ "id": "{create_task_id()}", "agent_name": "OrchestratorAgent", "action": "direct_answer", "details": {{"answer_prompt": "Formulate an answer for: {user_request}"}} }}

Example Plan:
[
    {{"id": "{create_task_id()}", "agent_name": "SearchAgent", "action": "perform_search", "details": {{"query": "current Python version"}}}},
    {{"id": "{create_task_id()}", "agent_name": "CodeAgent", "action": "generate_code", "details": {{"prompt": "Write a python script to parse search results and print the version.", "context_data": " risultati_ricerca "}}}},
    {{"id": "{create_task_id()}", "agent_name": "CodeAgent", "action": "execute_code", "details": {{"code": "use_generated_code"}}}}
]
Output only the JSON plan.
"""
    messages = [SystemMessage(content=prompt)]
    try:
        response = orchestrator_llm.invoke(messages)
        plan_str = response.content.strip()
        # Clean up potential markdown ```json ... ```
        if plan_str.startswith("```json"):
            plan_str = plan_str[len("```json"):].strip()
        if plan_str.endswith("```"):
            plan_str = plan_str[:-len("```")].strip()
        new_plan_tasks = json.loads(plan_str)
        state.plan = [Task(**task_data) for task_data in new_plan_tasks]  # Validate with Pydantic
        state.overall_status = "executing"
        state.next_node_to_call = "task_router" # Next, route to the first task
        print(f"AGENT: Orchestrator - Plan created: {state.plan}")
    except Exception as e:
        print(f"AGENT: Orchestrator - Error creating plan: {e}. LLM Output: {response.content if 'response' in locals() else 'N/A'}")
        state.error_message = f"Failed to create a plan: {e}"
        state.overall_status = "error"
        state.next_node_to_call = "final_response_node"  # Go to finalize with error
    return state


def orchestrator_direct_answer_node(state: AgentState) -> AgentState:
    """Handles the 'direct_answer' action for the Orchestrator."""
    print("AGENT: Orchestrator - Attempting direct answer...")
    current_task = next((task for task in state.plan if task.id == state.current_task_id), None)
    if not current_task or not current_task.details.get("answer_prompt"):
        state.error_message = "Orchestrator: Direct answer task details missing."
        current_task.status = "failed"
        current_task.error = state.error_message
        state.next_node_to_call = "task_router" # let router decide if plan continues or ends
        return state

    prompt_for_answer = current_task.details["answer_prompt"]
    # You might want to provide context from previous steps if any for a better direct answer
    # e.g., state.search_results
    
    messages = [
        SystemMessage(content="You are a helpful assistant. Answer the user's query based on the provided prompt."),
        HumanMessage(content=prompt_for_answer)
    ]
    try:
        response = orchestrator_llm.invoke(messages)
        answer = response.content.strip()
        current_task.result = {"answer": answer}
        current_task.status = "completed"
        state.final_response = answer  # A direct answer can be the final response
        print(f"AGENT: Orchestrator - Direct answer generated: {answer}")
    except Exception as e:
        print(f"AGENT: Orchestrator - Error generating direct answer: {e}")
        current_task.status = "failed"
        current_task.error = str(e)
        state.error_message = f"Failed to generate direct answer: {e}"

    state.next_node_to_call = "task_router" # Continue plan or finalize
    return state


def final_response_node(state: AgentState) -> AgentState:
    print("AGENT: Orchestrator - Synthesizing Final Response...")
    state.overall_status = "synthesizing"

    if state.final_response: # Already set by direct answer or critical error
        print(f"AGENT: Orchestrator - Final response already determined: {state.final_response}")
        state.overall_status = "finished"
        state.next_node_to_call = "__end__"
        return state

    if state.error_message and not state.plan: # Major planning error
        state.final_response = f"I encountered an error and could not complete your request: {state.error_message}"
        state.overall_status = "error"
        state.next_node_to_call = "__end__"
        return state

    # Summarize execution for the LLM
    execution_summary = "User Request: " + state.original_request + "\nExecution Log:\n"
    for task in state.executed_tasks_log:
        execution_summary += f"- Task ID {task.id} ({task.agent_name} - {task.action}): {task.status}\n"
        if task.status == "completed" and task.result:
            try:
                result_summary = json.dumps(task.result, indent=0, ensure_ascii=False)
                if len(result_summary) > 300: result_summary = result_summary[:297] + "..."
                execution_summary += f"  Result: {result_summary}\n"
            except Exception:
                execution_summary += "  Result: (Could not serialize result for summary)\n"
        elif task.error:
            execution_summary += f"  Error: {task.error}\n"

    # Add latest data not yet in executed_tasks_log
    if state.search_results and (not state.executed_tasks_log or state.executed_tasks_log[-1].action != "perform_search"):
        execution_summary += f"\nLatest Search Results:\n{json.dumps(state.search_results[:2], indent=0)}\n" # Sample
    if state.generated_code and (not state.executed_tasks_log or state.executed_tasks_log[-1].action != "generate_code"):
        execution_summary += f"\nLatest Generated Code:\n{state.generated_code[:200]}...\n"
    if state.code_execution_stdout and (not state.executed_tasks_log or state.executed_tasks_log[-1].action != "execute_code"):
        execution_summary += f"\nLatest Code Output:\n{state.code_execution_stdout[:200]}...\n"
    if state.error_message:
        execution_summary += f"\nOverall Error Message: {state.error_message}\n"

    synthesis_prompt = f"""Based on the user's request and the following execution summary, generate a comprehensive final response for the user.
If the goal was achieved, clearly state the outcome. If it failed, explain why. Be helpful and informative.
{execution_summary}
Final Response:
"""
    messages = [SystemMessage(content=synthesis_prompt)]
    try:
        response = orchestrator_llm.invoke(messages)
        state.final_response = response.content.strip()
        state.overall_status = "finished"
    except Exception as e:
        print(f"AGENT: Orchestrator - Error synthesizing final response: {e}")
        state.final_response = f"I encountered an issue while trying to formulate the final answer. Please review the execution log. Error: {e}"
        state.overall_status = "error"

    state.next_node_to_call = "__end__" # Mark completion of the graph
    return state


def search_agent_node(state: AgentState) -> AgentState:
    print("AGENT: SearchAgent - Performing Search...")
    current_task = next((task for task in state.plan if task.id == state.current_task_id), None)
    query = current_task.details.get("query")

    if not query:
        current_task.status = "failed"
        current_task.error = "No query provided for search."
        state.next_node_to_call = "task_router"
        return state

    state.search_queries.append(query)
    all_results_for_this_query = []
    tool_errors = {}
    found_good_results = False

    for tool_name in config.SEARCH_TOOL_PRIORITY:
        print(f"AGENT: SearchAgent - Trying tool: {tool_name}")
        search_function = SEARCH_FUNCTION_MAP.get(tool_name)
        if not search_function:
            print(f"AGENT: SearchAgent - Unknown search tool: {tool_name}")
            tool_errors[tool_name] = "Unknown search tool."
            continue
        
        try:
            results = search_function(query)
            if results and not any(res.get("error") for res in results): # Check if results are not errors themselves
                print(f"AGENT: SearchAgent - Found {len(results)} results with {tool_name}.")
                all_results_for_this_query.extend(results)
                state.last_search_tool_used = tool_name
                found_good_results = True # Found results, break and use them
                break 
            elif results and any(res.get("error") for res in results):
                print(f"AGENT: SearchAgent - Tool {tool_name} returned an error in its results: {results[0]['error']}")
                tool_errors[tool_name] = results[0]['error']
            else:
                print(f"AGENT: SearchAgent - No results from {tool_name}.")
        except Exception as e:
            print(f"AGENT: SearchAgent - Error with tool {tool_name}: {e}")
            tool_errors[tool_name] = str(e)
            
    state.search_results.extend(all_results_for_this_query) # Append to overall search results
    state.search_tool_errors.update(tool_errors)

    if found_good_results:
        current_task.status = "completed"
        current_task.result = {"search_results_summary": f"Found {len(all_results_for_this_query)} items using {state.last_search_tool_used}.", "data_preview": all_results_for_this_query[:2]}
    else:
        current_task.status = "failed"
        current_task.error = f"Search failed to find results. Errors: {json.dumps(tool_errors)}"
        state.error_message = current_task.error # Propagate error for overall status if needed

    state.next_node_to_call = "task_router"
    return state


def code_agent_installer_node(state: AgentState) -> AgentState:
    print("AGENT: CodeAgent - Installing Libraries...")
    current_task = next((task for task in state.plan if task.id == state.current_task_id), None)
    libraries = current_task.details.get("libraries", [])
    state.libraries_to_install = libraries # For visibility and potential use by execute step

    if not libraries:
        current_task.status = "completed" # Or "skipped"
        current_task.result = {"log": "No libraries specified for installation."}
        state.library_installation_status = "success" # Considered success if none to install
        state.next_node_to_call = "task_router"
        return state

    result = install_libraries(libraries) # This is the old way, see execute_python_code for new way
    # The new `execute_python_code` handles installation internally per execution.
    # So, this "install_libraries" action becomes more of a declaration for the subsequent "execute_code" step.
    # It doesn't need to actually *do* the install if the executor handles it.
    # For now, let's assume this step just validates and records the intent.

    state.library_installation_log = result["log"]
    if result["status"] == "success":
        current_task.status = "completed"
        current_task.result = result
        state.library_installation_status = "success"
    else:
        current_task.status = "failed"
        current_task.error = result["log"]
        state.library_installation_status = "failure"
        state.error_message = f"Library installation failed: {result['log'][:200]}..."

    state.next_node_to_call = "task_router"
    return state


def code_agent_generator_node(state: AgentState) -> AgentState:
    print("AGENT: CodeAgent - Generating Code...")
    current_task = \
        next((task for task in state.plan if task.id == state.current_task_id),
             None)
    prompt_for_code = current_task.details.get("prompt")
    context_data = current_task.details.get("context_data", "") # e.g., search results as string

    # Enhance context data if previous results exist
    if not context_data and state.search_results: # If no explicit context, but search results exist
        context_data = "Relevant search results:\n" + json.dumps(state.search_results, indent=2)

    if not prompt_for_code:
        current_task.status = "failed"
        current_task.error = "No prompt provided for code generation."
        state.next_node_to_call = "task_router"
        return state

    full_prompt = f"""Generate a Python script based on the following prompt.
The script should be runnable and self-contained, or use libraries that will be declared/installed.
Ensure the script prints its main output to STDOUT. If it produces files, mention that in comments.
Do not wrap the code in markdown backticks.

Prompt: {prompt_for_code}

Contextual Data (if any, use as needed):
{context_data if context_data else "No specific context data provided beyond the prompt."}

Python Code:
"""
    messages = [SystemMessage(content=full_prompt)]
    try:
        response = code_gen_llm.invoke(messages)
        generated_script = response.content.strip()
        state.generated_code = generated_script  # Store for potential 'use_generated_code'
        current_task.status = "completed"
        current_task.result = {"generated_code_preview": generated_script[:200] + "..."}  # Don't store full code in result directly
        print(f"AGENT: CodeAgent - Code Generated:\n{generated_script[:300]}...")
    except Exception as e:
        current_task.status = "failed"
        current_task.error = f"Code generation failed: {e}"
        state.error_message = current_task.error
        print(f"AGENT: CodeAgent - Code Generation Error: {e}")

    state.next_node_to_call = "task_router"
    return state


def code_agent_executor_node(state: AgentState) -> AgentState:
    print("AGENT: CodeAgent - Executing Code...")
    current_task = next((task for task in state.plan if task.id == state.current_task_id), None)
    code_input = current_task.details.get("code")
    # Libraries specifically noted as relevant for *this* execution.
    # These are the libraries that should have been "installed" (declared) by a previous 'install_libraries' task.
    relevant_libraries = current_task.details.get("relevant_libraries", state.libraries_to_install or [])

    script_to_run = ""
    if code_input == "use_generated_code":
        if state.generated_code:
            script_to_run = state.generated_code
        else:
            current_task.status = "failed"
            current_task.error = "Asked to execute generated code, but no code was previously generated."
            state.next_node_to_call = "task_router"
            return state
    elif isinstance(code_input, str):
        script_to_run = code_input
    else:
        current_task.status = "failed"
        current_task.error = "No valid code provided for execution."
        state.next_node_to_call = "task_router"
        return state

    # The new execute_python_code handles library installation internally for that run
    execution_result = execute_python_code(script_to_run, libraries_needed=relevant_libraries)

    state.code_execution_stdout = execution_result.get("stdout")
    state.code_execution_stderr = execution_result.get("stderr")

    if execution_result["status"] == "success":
        current_task.status = "completed"
        current_task.result = {"stdout": state.code_execution_stdout[:500], "stderr": state.code_execution_stderr[:500]} # Preview
        print(f"AGENT: CodeAgent - Execution Success. STDOUT:\n{state.code_execution_stdout[:300]}...")
        if state.code_execution_stderr: print(f"AGENT: CodeAgent - Execution STDERR:\n{state.code_execution_stderr[:300]}...")
    else:
        current_task.status = "failed"
        current_task.error = f"Execution failed. STDERR: {state.code_execution_stderr[:500]}"
        state.error_message = current_task.error
        print(f"AGENT: CodeAgent - Execution Failed. STDERR:\n{state.code_execution_stderr}")
        if state.code_execution_stdout: print(f"AGENT: CodeAgent - Execution STDOUT (on failure):\n{state.code_execution_stdout[:300]}...")

    state.next_node_to_call = "task_router"
    return state
