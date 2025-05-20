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
            # If a task failed, the orchestrator might need to re-plan or halt.
            # For now, we continue the plan unless a critical error is set on state.error_message.
            if task_just_processed.status == "failed" and not state.error_message:
                state.error_message = f"Task {task_just_processed.id} ({task_just_processed.action}) failed: {task_just_processed.error}"

    next_pending_task = next((task for task in state.plan if task.status == "pending"), None)

    if state.error_message and not next_pending_task: # If an error occurred and no more tasks, or major planning error
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
            state.next_node_to_call = "final_response_node" # Go to finalize with error
            if state.current_task_id: # Mark current task as failed
                next_pending_task.status = "failed"
                next_pending_task.error = state.error_message
                state.executed_tasks_log.append(next_pending_task.copy(deep=True))
    else:
        print("ROUTER: Plan complete. Proceeding to final response.")
        state.next_node_to_call = "final_response_node"
        state.current_task_id = None
    return state


def create_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("orchestrator_planner", orchestrator_planner_node)
    workflow.add_node("task_router", task_router_node)

    for agent, actions in AGENT_ACTION_MAP.items():
        for action, node_func in actions.items():
            workflow.add_node(f"{agent}_{action}", node_func)

    workflow.add_node("final_response_node", final_response_node)

    workflow.set_entry_point("orchestrator_planner")
    
    workflow.add_edge("orchestrator_planner", "task_router")

    def route_logic(state: AgentState):
        if state.next_node_to_call == "__end__":
            return END
        return state.next_node_to_call  # This will be the name of the next node to execute

    for agent, actions in AGENT_ACTION_MAP.items():
        for action in actions.keys():
            workflow.add_edge(f"{agent}_{action}", "task_router")
            
    workflow.add_conditional_edges(
        "task_router",
        route_logic,
        {
            **{f"{agent}_{action}": f"{agent}_{action}"
               for agent, actions in AGENT_ACTION_MAP.items()
               for action in actions.keys()},
            "final_response_node": "final_response_node",
            END: END
        }
    )

    workflow.add_edge("final_response_node", END)

    app = workflow.compile()
    return app
