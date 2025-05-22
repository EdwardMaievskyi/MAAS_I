
# MAAS-I: Multi-Agent Assistance System

This project implements a multi-agent system that can handle complex user requests by breaking them down into a series of coordinated tasks executed by specialized agents. GUI was created using Gradio and simplifies communication with agents for the end-user. Below you can find a summary of what has been achieved.

## Core Architecture

Agent-Based Architecture: The system uses multiple specialized agents
- OrchestratorAgent: Plans and coordinates tasks, provides direct answers for simple queries
- SearchAgent: Performs web searches using various search engines
- CodeAgent: Generates and executes Python code in an isolated Docker environment

State Management: Uses a Pydantic-based state model (AgentState) to track the execution state across the entire workflow.

Task Planning: The orchestrator dynamically creates a plan of tasks based on the user's request.

Workflow Engine: Uses LangGraph to create a directed graph of agent nodes with conditional routing.

## Key Features

Web Search Capabilities:
- Multiple search providers (DuckDuckGo, Tavily, BraveSearch, SerpAPI)
- Fallback mechanism if one search provider fails

Code Generation and Execution:
- Secure code execution in isolated Docker containers
- Dynamic library installation for code requirements
- Error handling for code execution failures

Robust Error Handling:
- Task-level error tracking
- Graceful degradation when components fail
- Detailed execution logs

Flexible Planning:
- The system can create different plans based on the complexity of the request
- Direct answers for simple questions
- Multi-step plans for complex tasks

Technical Implementation
- Docker Integration - Uses Docker for isolated code execution, with dynamic image building. (can be changed to using E2B instead)
- API Integrations - Connects to various search APIs (Tavily, SerpAPI, BraveSearch).
- LLM Integration - Uses OpenAI's models for orchestration and code generation.
- State Graph - Implements a directed graph for workflow management with conditional routing.

## Application Functionality
The application works as follows:
User inputs a request or question
The orchestrator analyzes the request and creates a plan of tasks
The task router directs each task to the appropriate agent
Agents execute their specialized tasks (search, code generation, code execution)
Results from each task are stored in the shared state
The final response node synthesizes all results into a coherent answer
The answer is presented to the user
The system can handle a wide range of requests, from simple questions that can be answered directly to complex tasks requiring multiple steps of search and code execution.

