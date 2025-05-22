from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Literal, Union
from enum import Enum


class Task(BaseModel):
    id: str
    agent_name: Literal["OrchestratorAgent", "SearchAgent", "CodeAgent", "FinancialDataAgent"]
    action: str
    details: Dict[str, Any] = Field(default_factory=dict)
    status: Literal["pending", "in_progress", "completed", "failed"] = "pending"
    result: Optional[Any] = None
    error: Optional[str] = None


class FinancialRequestType(Enum):
    """Enumeration of supported financial request types"""
    STOCK_PRICE = "stock_price"
    COMPANY_INFO = "company_info"
    FINANCIAL_RATIOS = "financial_ratios"
    MARKET_DATA = "market_data"
    NEWS_ANALYSIS = "news_analysis"
    PORTFOLIO_ANALYSIS = "portfolio_analysis"
    HISTORICAL_STOCK_DATA = "historical_stock_data"


class FinancialData(BaseModel):
    """Structure for individual financial metrics"""
    name: str = Field(description="Name of the financial metric")
    ticker: Optional[str] = Field(description="Ticker symbol of the stock or ETF")
    value: Union[float, str, None] = Field(description="Value of the metric")
    unit: Optional[str] = Field(default=None, description="Unit of measurement (e.g., 'USD', '%', 'millions')")
    series: Optional[List[Dict[str, Any]]] = Field(default_factory=list, description="Series of data points")
    news: Optional[List[Dict[str, Any]]] = Field(default_factory=list, description="News related to this metric or ticker")
    source: Optional[str] = Field(default=None, description="Data source for this metric")


class CompanyInfo(BaseModel):
    """Structured company information"""
    symbol: Optional[str] = Field(default=None, description="Stock ticker symbol")
    name: Optional[str] = Field(default=None, description="Company name")
    sector: Optional[str] = Field(default=None, description="Business sector")
    industry: Optional[str] = Field(default=None, description="Specific industry")
    market_cap: Optional[float] = Field(default=None, description="Market capitalization")
    description: Optional[str] = Field(default=None, description="Company business description")


class AgentState(BaseModel):
    original_request: str = Field(
        description="The raw user request as initially received"
    )
    plan: List[Task] = Field(default_factory=list)
    executed_tasks_log: List[Task] = Field(default_factory=list) # Log of completed/failed tasks

    # Data passed between agents
    current_task_id: Optional[str] = None
    data_for_current_task: Optional[Dict[str, Any]] = None # Holds specific inputs for the current agent

    # Search Agent specific state
    search_queries: List[str] = Field(default_factory=list)
    search_results: List[Dict[str, Any]] = Field(default_factory=list)
    last_search_tool_used: Optional[str] = None
    search_tool_errors: Dict[str, str] = Field(default_factory=dict)

    # Financial Data Retrieving Agent specific state
    financial_data_request: Optional[List[FinancialRequestType]] = \
        Field(default=None,
              description="Systematized well-defined list of financial data requested by user if user requests financial data")
    financial_data_result: Optional[List[FinancialData]] = \
        Field(default=None,
              description="Data structure for financial data results if user requests financial data    ")
    company_info_result: Optional[List[CompanyInfo]] = \
        Field(default=None,
              description="Data structure for detailed company information if user requests financial data")

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
