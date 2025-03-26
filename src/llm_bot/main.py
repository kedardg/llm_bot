#!/usr/bin/env python
"""
Main entry point for the LLM Bot application.
Handles command processing and execution with error handling and logging.
"""
import sys
import warnings
import json

from llm_bot.crew import LlmBot

# Suppress pysbd syntax warnings
warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

def run():
    """
    Run the command processing crew with enhanced error handling and logging.
    
    Processes user commands either from command line arguments or uses a default command.
    Outputs results to console and saves to response.json file.
    """
    # Get user command from command line argument or use default
    if len(sys.argv) > 1:
        user_command = " ".join(sys.argv[1:])
    else:
        user_command = "Move forward 5 feet and Rotate clockwise 100 degrees. also Move forward 15 centimeters"
    
    inputs = {
        'user_command': user_command
    }
    
    try:
        # Create crew instance with error tracking
        crew = LlmBot().crew()
        print(f"\n🤖 Processing command: {user_command}\n")
        
        # Execute with output validation
        result = crew.kickoff(inputs=inputs)
        
        try:
            # Try to read from file first
            with open('response.json', 'r') as response_file:
                response_data = json.load(response_file)
                print("\n✅ Response from file:")
                print(json.dumps(response_data, indent=2))
                
                # Verify all commands are included
                if 'responses' in response_data:
                    print(f"\n📋 Processed {len(response_data['responses'])} commands:")
                    for i, cmd in enumerate(response_data['responses'], 1):
                        cmd_type = cmd['command'] or 'Vision/Chat'
                        desc = cmd['description'][:50] + ('...' if len(cmd['description']) > 50 else '')
                        print(f"  {i}. {cmd_type}: {desc}")
                
        except Exception as file_error:
            # Fall back to the result directly
            print("\n⚠️ Could not read from file, using direct result:")
            if hasattr(result, 'model_dump_json'):
                result_json = json.loads(result.model_dump_json())
            else:
                result_json = result.dict() if hasattr(result, 'dict') else result
            
            print(json.dumps(result_json, indent=2))
            
            # Save validated result
            with open('response.json', 'w') as f:
                json.dump(result_json, f, indent=2)
            
    except Exception as e:
        print("\n❌ Error during execution:")
        print(f"  {str(e)}")
        if hasattr(e, '__traceback__'):
            import traceback
            print("\nDetailed error trace:")
            traceback.print_tb(e.__traceback__)
        sys.exit(1)

def train():
    """
    Train the crew for a specified number of iterations.
    
    Args:
        sys.argv[1]: Number of training iterations
        sys.argv[2]: Output filename for training data
    
    Raises:
        Exception: If an error occurs during training
    """
    inputs = {
        "user_command": "Move forward 5 feet and Rotate clockwise 100 degrees and tell me what you see. also Move forward 15 centimeters"
    }
    try:
        LlmBot().crew().train(n_iterations=int(sys.argv[1]), filename=sys.argv[2], inputs=inputs)
    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")

def replay():
    """
    Replay the crew execution from a specific task.
    
    Args:
        sys.argv[1]: Task ID to replay
    
    Raises:
        Exception: If an error occurs during replay
    """
    try:
        LlmBot().crew().replay(task_id=sys.argv[1])
    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")

def test():
    """
    Test the crew execution and return results.
    
    Args:
        sys.argv[1]: Number of test iterations
        sys.argv[2]: OpenAI model name to use for testing
    
    Raises:
        Exception: If an error occurs during testing
    """
    inputs = {
        "user_command": "Move forward 5 feet and Rotate clockwise 100 degrees and tell me what you see. also Move forward 15 centimeters"
    }
    try:
        LlmBot().crew().test(n_iterations=int(sys.argv[1]), openai_model_name=sys.argv[2], inputs=inputs)
    except Exception as e:
        raise Exception(f"An error occurred while testing the crew: {e}")

if __name__ == "__main__":
    run()