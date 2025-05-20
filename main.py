import json
from state_model import AgentState, Task
from graph_builder import create_graph
from tool_shed import IS_DOCKER_IMAGE_READY
import config


def run_assistant():
    if not config.OPENAI_API_KEY:
        print("Error: OPENAI_API_KEY not found. Please set it in your .env file.")
        return

    print("Multi-Agent Assistance System Initialized.")
    print("Docker image build status:", "Ready" if IS_DOCKER_IMAGE_READY else "Failed/Not Available")

    graph = create_graph()

    while True:
        user_request = input("\n🤖 Ed, what task can I help you with today? (type 'quit' to exit)\n> ")
        if user_request.lower() == 'quit':
            break
        if not user_request.strip():
            continue

        initial_state = AgentState(original_request=user_request)

        print("\nProcessing your request...")
        """
        for event in graph.stream(initial_state, {"recursion_limit": 150}):
            node_name = list(event.keys())[0]
            state_snapshot = event[node_name]
            print(f"\n--- After Node: {node_name} ---")
            # print(json.dumps(state_snapshot, indent=2, default=lambda o: o.__dict__ if hasattr(o, '__dict__') else str(o)))
            if isinstance(state_snapshot, AgentState):
                print(f"  Next node to call: {state_snapshot.next_node_to_call}")
                if state_snapshot.current_task_id:
                    print(f"  Current task ID: {state_snapshot.current_task_id}")
                if node_name == "final_response_node":
                    print(f"\n✅ Final Response:\n{state_snapshot.final_response}")
            else:
                print(state_snapshot)
        """

        final_result_state = graph.invoke(initial_state,
                                          {"recursion_limit": 50})

        print("\n--- Execution Complete ---")

        if not isinstance(final_result_state, AgentState):
            try:
                # Create a new AgentState from the dict
                if hasattr(final_result_state, "__dict__"):
                    final_result_state = AgentState(**final_result_state)
                elif isinstance(final_result_state, dict):
                    final_result_state = AgentState(**final_result_state)
                print(f"Converted result to AgentState object")
            except Exception as e:
                print(f"⚠️ Critical Error: Expected final state to be an AgentState object, but got {type(final_result_state)}.")
                print("   Raw final state:", final_result_state)
                # Attempt to gracefully show some info if it's a dict from a specific node perhaps
                if isinstance(final_result_state, dict):
                    final_response_from_dict = final_result_state.get('final_response', "Response not found in dict.")
                    error_message_from_dict = final_result_state.get('error_message', "Error message not found in dict.")
                    print(f"\n⚠️ (From Dict) Final Response Attempt:\n{final_response_from_dict}")
                    print(f"\n⚠️ (From Dict) Error Message Attempt:\n{error_message_from_dict}")
                continue # Skip to next user request

        # Now we can safely access attributes because it's an AgentState object
        if final_result_state.final_response:
            print(f"\n✅ Final Response from Assistant:\n{final_result_state.final_response}")
        else:
            print("\n⚠️ Assistant could not generate a final response (final_response attribute was empty or None).")
        
        if final_result_state.error_message and not final_result_state.final_response:
            print(f"\nOverall System Error: {final_result_state.error_message}")


if __name__ == "__main__":
    run_assistant()
