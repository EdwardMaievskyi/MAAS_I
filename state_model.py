from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Literal


class Task(BaseModel):
    id: str
    agent_name: Literal["OrchestratorAgent", "SearchAgent", "CodeAgent"]
    action: str
    details: Dict[str, Any] = Field(default_factory=dict)
    status: Literal["pending", "in_progress", "completed", "failed"] = "pending"
    result: Optional[Any] = None
    error: Optional[str] = None


class AgentState(BaseModel):
    original_request: str
    plan: List[Task] = Field(default_factory=list)
    executed_tasks_log: List[Task] = Field(default_factory=list) # Log of completed/failed tasks

    # Data passed between agents
    current_task_id: Optional[str] = None
    data_for_current_task: Optional[Dict[str, Any]] = None # Holds specific inputs for the current agent

    # Search Agent specific state
    search_queries: List[str] = Field(default_factory=list)
    search_results: List[Dict[str, Any]] = Field(default_factory=list) # Aggregated results
    last_search_tool_used: Optional[str] = None
    search_tool_errors: Dict[str, str] = Field(default_factory=dict)

    # Code Agent specific state
    code_to_execute: Optional[str] = None
    generated_code: Optional[str] = None
    libraries_to_install: List[str] = Field(default_factory=list)
    library_installation_log: Optional[str] = None
    library_installation_status: Optional[Literal["success", "failure"]] = None
    code_execution_stdout: Optional[str] = None
    code_execution_stderr: Optional[str] = None

    # Orchestrator state
    final_response: Optional[str] = None
    overall_status: Literal["planning", "executing", "synthesizing", "finished", "error"] = "planning"
    error_message: Optional[str] = None

    # For routing
    next_node_to_call: Optional[str] = None
