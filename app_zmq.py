"""
ZMQ server implementation for the LLM Bot.
Provides a ZeroMQ-based interface for processing commands and handling image data.
"""

import zmq
import json
import base64
import signal
import sys
from typing import Dict, Optional
from crew import LlmBot

class LLMBotServer:
    """
    ZMQ server that handles LLM Bot requests and responses.
    
    Attributes:
        port (int): Port number for the ZMQ server
        context (zmq.Context): ZMQ context
        socket (zmq.Socket): ZMQ REP socket
        running (bool): Server running state
        crew (LlmBot): LLM Bot crew instance
    """
    
    def __init__(self, port: int = 5555):
        """
        Initialize the ZMQ server.
        
        Args:
            port (int): Port number to listen on, defaults to 5555
        """
        self.port = port
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.running = True
        
        # Initialize crew instance once during startup
        print("Initializing LLM Bot crew...")
        self.crew = LlmBot().crew()
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

    def signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        print("\nShutting down server...")
        self.running = False
        self.socket.close()
        self.context.term()
        sys.exit(0)

    def process_request(self, data: Dict) -> Dict:
        """
        Process incoming request and return response.
        
        Args:
            data (Dict): Request data containing user_command and optional image
        
        Returns:
            Dict: Response containing status and result/error
        """
        try:
            # Extract user command and image if present
            user_command = data.get('user_command', '')
            image_base64 = data.get('image', None)
            
            inputs = {
                'user_command': user_command
            }
            
            # If image is present, decode it and add to inputs
            if image_base64:
                try:
                    # Remove data URL prefix if present
                    if ',' in image_base64:
                        image_base64 = image_base64.split(',')[1]
                    image_bytes = base64.b64decode(image_base64)
                    inputs['image'] = image_bytes
                except Exception as e:
                    return {
                        'status': 'error',
                        'error': f'Invalid image data: {str(e)}'
                    }
            
            # Use the existing crew instance
            result = self.crew.kickoff(inputs=inputs)
            
            # Convert result to JSON-serializable format
            if hasattr(result, 'model_dump_json'):
                result_json = json.loads(result.model_dump_json())
            else:
                result_json = result.dict() if hasattr(result, 'dict') else result
            
            return {
                'status': 'success',
                'result': result_json
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }

    def run(self):
        """
        Start the server and handle incoming requests.
        
        Continuously listens for requests and processes them until shutdown.
        Handles various error conditions and provides appropriate responses.
        """
        try:
            self.socket.bind(f"tcp://*:{self.port}")
            print(f"Server started on port {self.port}")
            
            while self.running:
                try:
                    # Wait for next request from client
                    message = self.socket.recv_json()
                    print(f"Received request: {message.get('user_command', '')[:50]}...")
                    
                    # Process the request
                    response = self.process_request(message)
                    
                    # Send reply back to client
                    self.socket.send_json(response)
                    
                except zmq.ZMQError as e:
                    if self.running:  # Only log error if we're still meant to be running
                        print(f"ZMQ Error: {e}")
                except json.JSONDecodeError:
                    error_response = {
                        'status': 'error',
                        'error': 'Invalid JSON format'
                    }
                    self.socket.send_json(error_response)
                except Exception as e:
                    error_response = {
                        'status': 'error',
                        'error': str(e)
                    }
                    self.socket.send_json(error_response)
                    
        finally:
            self.socket.close()
            self.context.term()

if __name__ == "__main__":
    server = LLMBotServer()
    server.run()