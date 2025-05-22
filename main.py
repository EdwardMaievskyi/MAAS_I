import gradio as gr
import threading
import queue
import time
from typing import List, Dict, Any, Optional, Generator

from state_model import AgentState
from graph_builder import create_graph
from tool_shed import IS_DOCKER_IMAGE_READY
import config


class AssistantBackend:
    def __init__(self):
        """Initialize the assistant backend."""
        self.graph = create_graph()
        self.is_ready = config.OPENAI_API_KEY is not None and IS_DOCKER_IMAGE_READY

    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the assistant."""
        return {
            "api_key_configured": config.OPENAI_API_KEY is not None,
            "docker_ready": IS_DOCKER_IMAGE_READY,
            "is_ready": self.is_ready
        }

    def process_request(self, user_request: str) -> Generator[str, None, AgentState]:
        """Process a user request and yield updates as they happen.

        Args:
            user_request (str): The user's request to be processed.

        Returns:
            Generator[str, None, AgentState]: A generator that yields updates as the request is processed.
        """
        if not self.is_ready:
            yield "⚠️ System not fully initialized. Please check API keys and Docker status."
            return None

        if not user_request.strip():
            yield "Please enter a valid request."
            return None

        initial_state = AgentState(original_request=user_request)

        yield "Processing your request...\n"

        # Stream the execution with updates
        for event in self.graph.stream(initial_state, {"recursion_limit": 50}):
            node_name = list(event.keys())[0]
            state_snapshot = event[node_name]

            update = f"Step: {node_name}"
            if isinstance(state_snapshot, AgentState) and state_snapshot.current_task_id:
                current_task = next((t for t in state_snapshot.plan if t.id == state_snapshot.current_task_id), None)
                if current_task:
                    update += f" - {current_task.agent_name}: {current_task.action}"

            yield update + "\n"
            time.sleep(0.1)

        # Get the final result
        final_result_state = self.graph.invoke(initial_state, {"recursion_limit": 50})

        # Convert to AgentState if needed
        if not isinstance(final_result_state, AgentState):
            try:
                if hasattr(final_result_state, "__dict__"):
                    final_result_state = AgentState(**final_result_state)
                elif isinstance(final_result_state, dict):
                    final_result_state = AgentState(**final_result_state)
            except Exception as e:
                yield f"⚠️ Error processing result: {str(e)}"
                return None

        # Return the final response
        if final_result_state.final_response:
            yield final_result_state.final_response
        elif final_result_state.error_message:
            yield f"⚠️ Error: {final_result_state.error_message}"
        else:
            yield "⚠️ No response generated."

        return final_result_state


class ChatInterface:
    def __init__(self):
        """Initialize the chat interface."""
        self.backend = AssistantBackend()
        self.message_queue = queue.Queue()
        self.current_state = None

    def check_status(self) -> str:
        """Check and return the status of the assistant."""
        status = self.backend.get_status()
        if status["is_ready"]:
            return "✅ Assistant is ready to help!"
        else:
            issues = []
            if not status["api_key_configured"]:
                issues.append("- OpenAI API key not configured")
            if not status["docker_ready"]:
                issues.append("- Docker environment not ready")
            return "⚠️ Assistant not fully initialized:\n" + "\n".join(issues)

    def process_message(self, message: str, history: List[List[str]]) -> Generator[List[List[str]], None, None]:
        """Process a user message and yield updates for the UI.

        Args:
            message (str): The user's message to be processed.
            history (List[List[str]]): The chat history.

        Returns:
            Generator[List[List[str]], None, None]: A generator that yields updates for the UI.
        """
        if not message.strip():
            yield history + [["", "Please enter a valid message."]]
            return

        # Create a new history with the user's message
        new_history = history + [[message, "Thinking..."]]
        yield new_history

        # Process the request in the background
        response_generator = self.backend.process_request(message)

        # Stream the response
        last_response = "Thinking..."
        for update in response_generator:
            last_response = update
            new_history[-1][1] = last_response
            yield new_history

        # Store the final state if it was returned
        if isinstance(response_generator, AgentState):
            self.current_state = response_generator

    def create_interface(self) -> gr.Blocks:
        """Create and return the Gradio interface."""
        # Custom CSS to apply Roboto font throughout the interface
        custom_css = """
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');

        * {
            font-family: 'Roboto', sans-serif !important;
        }

        .message {
            font-family: 'Roboto', sans-serif !important;
            font-size: 16px !important;
        }

        .message-bubble {
            font-family: 'Roboto', sans-serif !important;
        }

        .prose {
            font-family: 'Roboto', sans-serif !important;
        }

        code, pre {
            font-family: 'Roboto Mono', monospace !important;
        }
        """

        with gr.Blocks(title="Multi-Agent Assistant", theme=gr.themes.Soft(), css=custom_css) as interface:
            gr.Markdown("# 🤖 Multi-Agent Assistant")

            with gr.Row():
                status_box = gr.Textbox(
                    value=self.check_status(),
                    label="System Status",
                    interactive=False
                )
                refresh_btn = gr.Button("Refresh Status")

            chatbot = gr.Chatbot(
                height=500,
                show_copy_button=True,
                bubble_full_width=False,
                show_label=False
            )

            with gr.Row():
                msg_box = gr.Textbox(
                    placeholder="Ask me anything...",
                    label="Your Request",
                    scale=9
                )
                submit_btn = gr.Button("Send", scale=1)

            # Add examples
            gr.Examples(
                examples=[
                    "What is an average price of BTC in the last 30 days?",
                    "Write a Python script to analyze sentiment in tweets",
                    "Find information about the latest AI research and summarize it",
                    "Create a data visualization of global CO2 emissions"
                ],
                inputs=msg_box
            )

            # Event handlers
            refresh_btn.click(
                fn=self.check_status,
                outputs=status_box
            )

            msg_box.submit(
                fn=self.process_message,
                inputs=[msg_box, chatbot],
                outputs=[chatbot],
                queue=True
            ).then(
                fn=lambda: "",
                outputs=msg_box
            )

            submit_btn.click(
                fn=self.process_message,
                inputs=[msg_box, chatbot],
                outputs=[chatbot],
                queue=True
            ).then(
                fn=lambda: "",
                outputs=msg_box
            )

            # Initialize
            interface.load(
                fn=self.check_status,
                outputs=status_box
            )

        return interface


def launch_ui():
    """Create and launch the UI."""
    chat_interface = ChatInterface()
    interface = chat_interface.create_interface()
    interface.queue()
    interface.launch(share=False)


if __name__ == "__main__":
    launch_ui()
