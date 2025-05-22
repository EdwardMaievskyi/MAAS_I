from langgraph.graph import StateGraph, END
from state_model import AgentState
from agents import (
    orchestrator_planner_node,
    orchestrator_direct_answer_node,
    search_agent_node,
    code_agent_installer_node,
    code_agent_generator_node,
    code_agent_executor_node,
    final_response_node
)

# Mapping actions to agent node functions
AGENT_ACTION_MAP = {
    "OrchestratorAgent": {
        "direct_answer": orchestrator_direct_answer_node,
    },
    "SearchAgent": {
        "perform_search": search_agent_node,
    },
    "CodeAgent": {
        "install_libraries": code_agent_installer_node,
        "generate_code": code_agent_generator_node,
        "execute_code": code_agent_executor_node,
    }
}


def task_router_node(state: AgentState) -> AgentState:
    """Routes to the next task in the plan or to finalization."""
    print("ROUTER: Determining next step...")

    # Log the completed/failed task
    if state.current_task_id:
        task_just_processed = next((task for task in state.plan if task.id == state.current_task_id), None)
        if task_just_processed:
            state.executed_tasks_log.append(task_just_processed.model_copy(deep=True))
            # If a task failed, analyze the error and decide if re-planning is needed
            if task_just_processed.status == "failed":
                print(f"ROUTER: Task {task_just_processed.id} ({task_just_processed.action}) failed: {task_just_processed.error}")
                
                # Determine if this error requires re-planning
                critical_error_keywords = ["permission denied", "not found", "invalid input", "missing data"]
                is_critical_error = any(keyword in task_just_processed.error.lower() for keyword in critical_error_keywords) if task_just_processed.error else False
                
                # Check if this is a search or code generation failure that might benefit from re-planning
                is_recoverable_failure = (
                    (task_just_processed.agent_name == "SearchAgent" and task_just_processed.action == "perform_search") or
                    (task_just_processed.agent_name == "CodeAgent" and task_just_processed.action in ["generate_code", "execute_code"])
                )
                
                # If this is a critical error or a recoverable failure, consider re-planning
                if is_critical_error or is_recoverable_failure:
                    print("ROUTER: Critical or recoverable error detected. Triggering re-planning.")
                    state.error_message = f"Task {task_just_processed.id} failed: {task_just_processed.error}. Re-planning required."
                    state.next_node_to_call = "orchestrator_planner"  # Go back to planning
                    return state
                else:
                    # Non-critical error, continue with plan but note the error
                    state.error_message = f"Task {task_just_processed.id} ({task_just_processed.action}) failed: {task_just_processed.error}"

    next_pending_task = next((task for task in state.plan if task.status == "pending"), None)

    # If we have an error message but still have pending tasks, assess if we should continue or re-plan
    if state.error_message and next_pending_task:
        # Check if we've had multiple failures that suggest we need a new approach
        failed_tasks_count = sum(1 for task in state.executed_tasks_log if task.status == "failed")
        if failed_tasks_count >= 2:  # If we've had multiple failures, consider re-planning
            print(f"ROUTER: Multiple failures detected ({failed_tasks_count}). Triggering re-planning.")
            state.next_node_to_call = "orchestrator_planner"
            return state

    if state.error_message and not next_pending_task:  # If an error occurred and no more tasks
        print(f"ROUTER: Error occurred ('{state.error_message}') or plan ended with error. Proceeding to final response.")
        state.next_node_to_call = "final_response_node"
        return state

    if next_pending_task:
        state.current_task_id = next_pending_task.id
        next_pending_task.status = "in_progress"

        agent_name = next_pending_task.agent_name
        action_name = next_pending_task.action

        if agent_name in AGENT_ACTION_MAP and action_name in AGENT_ACTION_MAP[agent_name]:
            state.next_node_to_call = f"{agent_name}_{action_name}"
            print(f"ROUTER: Routing to task ID {state.current_task_id} -> Node '{state.next_node_to_call}'")
        else:
            print(f"ROUTER: Unknown agent/action: {agent_name}/{action_name}. Ending.")
            state.error_message = f"Router error: Unknown agent or action '{agent_name}/{action_name}'."
            state.next_node_to_call = "final_response_node"  # Go to finalize with error
            if state.current_task_id:
                next_pending_task.status = "failed"
                next_pending_task.error = state.error_message
                state.executed_tasks_log.append(next_pending_task.model_copy(deep=True))
    else:
        print("ROUTER: Plan complete. Proceeding to final response.")
        state.next_node_to_call = "final_response_node"
        state.current_task_id = None

    # At the beginning of the function
    valid_destinations = list(AGENT_ACTION_MAP.keys()) + ["orchestrator_planner", "final_response_node"]
    
    # Before returning state with next_node_to_call
    if state.next_node_to_call not in valid_destinations and state.next_node_to_call != "__end__":
        print(f"ROUTER WARNING: Invalid destination '{state.next_node_to_call}'. Defaulting to final_response_node.")
        state.error_message = f"Routing error: Invalid destination '{state.next_node_to_call}'."
        state.next_node_to_call = "final_response_node"
    
    return state


def create_graph():
    workflow = StateGraph(AgentState)

    # Add all nodes
    workflow.add_node("orchestrator_planner", orchestrator_planner_node)
    workflow.add_node("task_router", task_router_node)
    workflow.add_node("final_response_node", final_response_node)

    for agent, actions in AGENT_ACTION_MAP.items():
        for action, node_func in actions.items():
            workflow.add_node(f"{agent}_{action}", node_func)

    # Set entry point
    workflow.set_entry_point("orchestrator_planner")

    # Add basic edges
    workflow.add_edge("orchestrator_planner", "task_router")
    
    # Add edges from agent nodes back to router
    for agent, actions in AGENT_ACTION_MAP.items():
        for action in actions.keys():
            workflow.add_edge(f"{agent}_{action}", "task_router")

    # Define conditional routing logic
    def route_logic(state: AgentState):
        destination = state.next_node_to_call if state.next_node_to_call != "__end__" else END
        print(f"ROUTER DEBUG: Routing from 'task_router' to '{destination}'")
        return destination

    # Add conditional edges from router to all possible destinations
    workflow.add_conditional_edges(
        "task_router",
        route_logic,
        {
            # Map all agent actions
            **{f"{agent}_{action}": f"{agent}_{action}"
               for agent, actions in AGENT_ACTION_MAP.items()
               for action in actions.keys()},
            # Add specific nodes
            "orchestrator_planner": "orchestrator_planner",  # This is the key fix
            "final_response_node": "final_response_node",
            END: END
        }
    )

    # Add final edge
    workflow.add_edge("final_response_node", END)

    # Compile and return
    app = workflow.compile()
    return app
