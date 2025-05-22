import json
import uuid
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from typing import List, Dict
import pandas as pd

import config
from state_model import AgentState, Task, FinancialData, CompanyInfo
from tool_shed import SEARCH_FUNCTION_MAP, \
    execute_python_code, install_libraries

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
3.  FinancialDataAgent:
    -   action: "retrieve_stock_prices"
        details: {{"request_type": "stock_price", "tickers": ["AAPL", "MSFT"], "time_period": {{"start": "2023-01-01", "end": "2023-12-31"}}}}
        (Retrieves current or historical stock prices for specified tickers.)
    -   action: "retrieve_company_info"
        details: {{"request_type": "company_info", "tickers": ["AAPL", "MSFT"]}}
        (Retrieves company information like sector, industry, and description.)
    -   action: "retrieve_financial_ratios"
        details: {{"request_type": "financial_ratios", "tickers": ["AAPL"], "metrics": ["pe_ratio", "dividend_yield"]}}
        (Retrieves financial ratios and metrics. Available metrics: pe_ratio, forward_pe, peg_ratio, price_to_book, dividend_yield, profit_margin, return_on_equity, beta, eps.)
    -   action: "retrieve_market_data"
        details: {{"request_type": "market_data", "metrics": ["gdp", "inflation"], "time_period": {{"start": "2020-01-01"}}}}
        (Retrieves market-wide economic data. Available metrics: gdp, inflation, unemployment, interest_rate, consumer_sentiment, retail_sales, industrial_production, housing_starts.)
    -   action: "retrieve_news"
        details: {{"request_type": "news_analysis", "tickers": ["AAPL"]}}
        (Retrieves and analyzes recent news for specified tickers.)
    -   action: "retrieve_historical_data"
        details: {{"request_type": "historical_stock_data", "tickers": ["AAPL"], "time_period": {{"start": "2023-01-01", "end": "2023-12-31"}}}}
        (Retrieves detailed historical data for analysis, including open, high, low, close prices and volume.)

User Request: "{user_request}"
{previous_steps_summary}

Based on the request and any previous errors, create a JSON list of tasks. Each task should have "id", "agent_name", "action", and "details".
Ensure IDs are unique for new tasks.
Not all agents and actions are needed for every request. Plan and optimize appropriately. Feel free to combine information from multiple agents if needed.
If the user request is a simple question that can be answered directly, you can create a plan with a single task for yourself:
    {{ "id": "{create_task_id()}", "agent_name": "OrchestratorAgent", "action": "direct_answer", "details": {{"answer_prompt": "Formulate an answer for: {user_request}"}} }}

Example Plan:
[
    {{"id": "{create_task_id()}", "agent_name": "SearchAgent", "action": "perform_search", "details": {{"query": "current Python version"}}}},
    {{"id": "{create_task_id()}", "agent_name": "CodeAgent", "action": "generate_code", "details": {{"prompt": "Write a python script to parse search results and print the version.", "context_data": " risultati_ricerca "}}}},
    {{"id": "{create_task_id()}", "agent_name": "CodeAgent", "action": "execute_code", "details": {{"code": "use_generated_code"}}}}
]

For financial requests, use the FinancialDataAgent. Example:
[
    {{"id": "{create_task_id()}", "agent_name": "FinancialDataAgent", "action": "retrieve_stock_prices", "details": {{"request_type": "stock_price", "tickers": ["AAPL"], "time_period": {{"start": "2023-01-01", "end": "2023-12-31"}}}}}},
    {{"id": "{create_task_id()}", "agent_name": "FinancialDataAgent", "action": "retrieve_company_info", "details": {{"request_type": "company_info", "tickers": ["AAPL"]}}}}
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


def financial_data_agent_node(state: AgentState) -> AgentState:
    """Retrieves and analyzes financial data based on the current task."""
    print("AGENT: FinancialDataAgent - Retrieving financial data...")
    current_task = next((task for task in state.plan if task.id == state.current_task_id), None)
    
    if not current_task:
        state.error_message = "FinancialDataAgent: No current task found."
        state.next_node_to_call = "task_router"
        return state
    
    # Extract request details
    request_type = current_task.details.get("request_type")
    ticker_symbols = current_task.details.get("tickers", [])
    time_period = current_task.details.get("time_period", {})
    metrics = current_task.details.get("metrics", [])
    
    if not request_type:
        current_task.status = "failed"
        current_task.error = "No financial data request type specified."
        state.next_node_to_call = "task_router"
        return state
    
    # Initialize results containers if not already present
    if state.financial_data_result is None:
        state.financial_data_result = []
    if state.company_info_result is None:
        state.company_info_result = []
    
    # Process based on request type
    try:
        if request_type == "stock_price":
            _retrieve_stock_prices(state, ticker_symbols, time_period)
        elif request_type == "company_info":
            _retrieve_company_info(state, ticker_symbols)
        elif request_type == "financial_ratios":
            _retrieve_financial_ratios(state, ticker_symbols, metrics)
        elif request_type == "market_data":
            _retrieve_market_data(state, metrics, time_period)
        elif request_type == "news_analysis":
            _retrieve_and_analyze_news(state, ticker_symbols)
        elif request_type == "historical_stock_data":
            _retrieve_historical_data(state, ticker_symbols, time_period)
        else:
            current_task.status = "failed"
            current_task.error = f"Unknown financial data request type: {request_type}"
            state.next_node_to_call = "task_router"
            return state
        
        # If we got here, the request was processed
        current_task.status = "completed"
        
        # Create a summary of the data retrieved
        summary = _create_financial_data_summary(state)
        current_task.result = {"summary": summary}
        
        print(f"AGENT: FinancialDataAgent - Successfully retrieved {request_type} data")
    except Exception as e:
        import traceback
        traceback.print_exc()  # Print the full stack trace for debugging
        current_task.status = "failed"
        current_task.error = f"Error retrieving financial data: {str(e)}"
        print(f"AGENT: FinancialDataAgent - Error: {e}")
    
    state.next_node_to_call = "task_router"
    return state


def _retrieve_stock_prices(state: AgentState, tickers: List[str], time_period: Dict[str, str]) -> None:
    """Retrieves current or historical stock prices."""
    from tool_shed import get_historical_yahoo_finance_stock_data
    
    start_date = time_period.get("start", None)
    end_date = time_period.get("end", None)
    
    for ticker in tickers:
        try:
            data = get_historical_yahoo_finance_stock_data(ticker, start_date, end_date)
            
            # Check if we got an error
            if isinstance(data, list) and len(data) > 0 and "error" in data[0]:
                raise Exception(data[0]["error"])
            
            # Process the data into our standard format
            if "Close" in data:
                close_prices = data["Close"]
                dates = list(close_prices.keys())
                values = list(close_prices.values())
                
                # Create time series data
                series_data = [{"date": str(date), "value": value} for date, value in zip(dates, values)]
                
                # Calculate average if we have data
                avg_price = sum(values) / len(values) if values else None
                
                # Add to financial data results
                state.financial_data_result.append(
                    FinancialData(
                        name=f"{ticker} Stock Price",
                        ticker=ticker,
                        value=avg_price,
                        unit="USD",
                        series=series_data,
                        source="Yahoo Finance"
                    )
                )
        except Exception as e:
            print(f"Error retrieving stock price for {ticker}: {e}")
            # Add error entry
            state.financial_data_result.append(
                FinancialData(
                    name=f"{ticker} Stock Price",
                    ticker=ticker,
                    value=None,
                    unit="USD",
                    source="Yahoo Finance",
                    series=[]
                )
            )


def _retrieve_company_info(state: AgentState, tickers: List[str]) -> None:
    """Retrieves company information for the specified tickers."""
    import yfinance as yf
    
    for ticker in tickers:
        try:
            ticker_obj = yf.Ticker(ticker)
            info = ticker_obj.info
            
            # Create company info object
            company_info = CompanyInfo(
                symbol=ticker,
                name=info.get("shortName", info.get("longName", ticker)),
                sector=info.get("sector"),
                industry=info.get("industry"),
                market_cap=info.get("marketCap"),
                description=info.get("longBusinessSummary")
            )
            
            state.company_info_result.append(company_info)
            
            # Also add a financial data entry for market cap
            if info.get("marketCap"):
                state.financial_data_result.append(
                    FinancialData(
                        name=f"{ticker} Market Cap",
                        ticker=ticker,
                        value=info.get("marketCap"),
                        unit="USD",
                        source="Yahoo Finance"
                    )
                )
        except Exception as e:
            print(f"Error retrieving company info for {ticker}: {e}")
            # Add empty company info
            state.company_info_result.append(
                CompanyInfo(
                    symbol=ticker,
                    name=ticker
                )
            )


def _retrieve_financial_ratios(state: AgentState, tickers: List[str], metrics: List[str]) -> None:
    """Retrieves financial ratios for the specified tickers."""
    import yfinance as yf
    
    # Map of metric names to their keys in yfinance
    metric_map = {
        "pe_ratio": "trailingPE",
        "forward_pe": "forwardPE",
        "peg_ratio": "pegRatio",
        "price_to_book": "priceToBook",
        "dividend_yield": "dividendYield",
        "profit_margin": "profitMargins",
        "return_on_equity": "returnOnEquity",
        "beta": "beta",
        "eps": "trailingEps"
    }
    
    # If no specific metrics requested, use all
    if not metrics:
        metrics = list(metric_map.keys())
    
    for ticker in tickers:
        try:
            ticker_obj = yf.Ticker(ticker)
            info = ticker_obj.info
            
            for metric in metrics:
                yf_key = metric_map.get(metric)
                if not yf_key or yf_key not in info:
                    continue
                
                value = info[yf_key]
                unit = "%" if metric in ["dividend_yield", "profit_margin", "return_on_equity"] else ""
                
                state.financial_data_result.append(
                    FinancialData(
                        name=f"{ticker} {metric.replace('_', ' ').title()}",
                        ticker=ticker,
                        value=value,
                        unit=unit,
                        source="Yahoo Finance"
                    )
                )
        except Exception as e:
            print(f"Error retrieving financial ratios for {ticker}: {e}")


def _retrieve_market_data(state: AgentState, metrics: List[str], time_period: Dict[str, str]) -> None:
    """Retrieves market-wide economic data."""
    from tool_shed import get_fred_index_data
    
    # Map of common economic indicators to their FRED codes
    fred_indicators = {
        "gdp": "GDP",
        "inflation": "CPIAUCSL",
        "unemployment": "UNRATE",
        "interest_rate": "FEDFUNDS",
        "consumer_sentiment": "UMCSENT",
        "retail_sales": "RSXFS",
        "industrial_production": "INDPRO",
        "housing_starts": "HOUST"
    }
    
    # If no specific metrics requested, use a default set
    if not metrics:
        metrics = ["gdp", "inflation", "unemployment", "interest_rate"]
    
    start_date = time_period.get("start", "2020-01-01")
    end_date = time_period.get("end", None)
    
    for metric in metrics:
        if metric not in fred_indicators:
            continue
            
        fred_code = fred_indicators[metric]
        try:
            data = get_fred_index_data(fred_code, start_date, end_date)
            
            # Check if we got an error
            if isinstance(data, list) and len(data) > 0 and "error" in data[0]:
                raise Exception(data[0]["error"])
            
            # Process the data
            dates = list(data.keys())
            values = list(data.values())
            
            # Create time series data
            series_data = [{"date": str(date), "value": value} for date, value in zip(dates, values)]
            
            # Calculate latest value
            latest_value = values[-1] if values else None
            
            # Determine unit based on metric
            unit = "%" if metric in ["inflation", "unemployment", "interest_rate"] else ""
            
            # Add to financial data results
            state.financial_data_result.append(
                FinancialData(
                    name=f"{metric.replace('_', ' ').title()}",
                    value=latest_value,
                    unit=unit,
                    series=series_data,
                    source="FRED"
                )
            )
        except Exception as e:
            print(f"Error retrieving {metric} data: {e}")


def _retrieve_and_analyze_news(state: AgentState, tickers: List[str]) -> None:
    """Retrieves and analyzes news for the specified tickers."""
    from tool_shed import get_yahoo_finance_news
    
    for ticker in tickers:
        try:
            news_data = get_yahoo_finance_news(ticker)
            
            # Check if we got an error
            if isinstance(news_data, list) and len(news_data) > 0 and "error" in news_data[0]:
                raise Exception(news_data[0]["error"])
            
            # Process news data
            news_items = []
            for item in news_data:
                news_items.append({
                    "title": item.get("title", ""),
                    "link": item.get("link", ""),
                    "publisher": item.get("publisher", ""),
                    "published_date": item.get("published_date", "")
                })
            
            # Add to financial data results
            state.financial_data_result.append(
                FinancialData(
                    name=f"{ticker} News",
                    ticker=ticker,
                    news=news_items,
                    source="Yahoo Finance News"
                )
            )
        except Exception as e:
            print(f"Error retrieving news for {ticker}: {e}")


def _retrieve_historical_data(state: AgentState, tickers: List[str], time_period: Dict[str, str]) -> None:
    """Retrieves detailed historical data for analysis."""
    # This is similar to _retrieve_stock_prices but with more metrics
    from tool_shed import get_historical_yahoo_finance_stock_data
    
    start_date = time_period.get("start", None)
    end_date = time_period.get("end", None)
    
    for ticker in tickers:
        try:
            data = get_historical_yahoo_finance_stock_data(ticker, start_date, end_date)
            
            # Check if we got an error
            if isinstance(data, list) and len(data) > 0 and "error" in data[0]:
                raise Exception(data[0]["error"])
            
            # Process the data for multiple metrics
            metrics = ["Open", "High", "Low", "Close", "Volume"]
            
            for metric in metrics:
                if metric not in data:
                    continue
                    
                metric_data = data[metric]
                dates = list(metric_data.keys())
                values = list(metric_data.values())
                
                # Create time series data
                series_data = [{"date": str(date), "value": value} for date, value in zip(dates, values)]
                
                # Calculate average
                avg_value = sum(values) / len(values) if values else None
                
                # Determine unit
                unit = "USD" if metric != "Volume" else "Shares"
                
                # Add to financial data results
                state.financial_data_result.append(
                    FinancialData(
                        name=f"{ticker} {metric}",
                        ticker=ticker,
                        value=avg_value,
                        unit=unit,
                        series=series_data,
                        source="Yahoo Finance"
                    )
                )
        except Exception as e:
            print(f"Error retrieving historical data for {ticker}: {e}")


def _create_financial_data_summary(state: AgentState) -> str:
    """Creates a summary of the financial data retrieved."""
    summary = []
    
    # Summarize company info
    if state.company_info_result:
        summary.append(f"Retrieved company information for {len(state.company_info_result)} companies.")
        for company in state.company_info_result:
            if company.name and company.symbol:
                summary.append(f"- {company.name} ({company.symbol})")
                if company.sector:
                    summary.append(f"  Sector: {company.sector}")
                if company.industry:
                    summary.append(f"  Industry: {company.industry}")
    
    # Summarize financial data
    if state.financial_data_result:
        data_by_ticker = {}
        for data in state.financial_data_result:
            ticker = data.ticker if data.ticker else "Market"
            if ticker not in data_by_ticker:
                data_by_ticker[ticker] = []
            data_by_ticker[ticker].append(data)
        
        summary.append(f"\nRetrieved {len(state.financial_data_result)} financial metrics:")
        for ticker, data_list in data_by_ticker.items():
            summary.append(f"- {ticker}:")
            for data in data_list:
                value_str = f"{data.value} {data.unit}" if data.value is not None else "N/A"
                summary.append(f"  • {data.name}: {value_str}")
                if data.series:
                    summary.append(f"    (Time series with {len(data.series)} data points)")
                if data.news:
                    summary.append(f"    (Retrieved {len(data.news)} news articles)")
    
    return "\n".join(summary)
