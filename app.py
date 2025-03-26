"""
FastAPI WebSocket server implementation for the LLM Bot.
Provides a WebSocket interface for real-time command processing and image handling.
"""

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import json
import base64
from typing import Dict, Optional
from crew import LlmBot

# Initialize FastAPI app
app = FastAPI()

# Configure CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Note: Configure specific origins for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time communication.
    
    Handles:
    - Command processing
    - Image data processing
    - Error handling and response formatting
    
    Args:
        websocket (WebSocket): WebSocket connection instance
    """
    await websocket.accept()
    
    try:
        while True:
            # Receive JSON data from client
            data = await websocket.receive_json()
            
            # Extract user command and image if present
            user_command = data.get('user_command', '')
            image_base64 = data.get('image', None)
            
            # Process the command using LlmBot
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
                    await websocket.send_json({
                        'error': f'Invalid image data: {str(e)}'
                    })
                    continue
            
            try:
                # Create crew instance and process command
                crew = LlmBot().crew()
                result = crew.kickoff(inputs=inputs)
                
                # Convert result to JSON
                if hasattr(result, 'model_dump_json'):
                    result_json = json.loads(result.model_dump_json())
                else:
                    result_json = result.dict() if hasattr(result, 'dict') else result
                
                # Send response back to client
                await websocket.send_json({
                    'status': 'success',
                    'result': result_json
                })
                
            except Exception as e:
                await websocket.send_json({
                    'status': 'error',
                    'error': str(e)
                })
                
    except Exception as e:
        await websocket.close(code=1001, reason=str(e))

@app.get("/")
async def root():
    """Root endpoint to verify server status."""
    return {"message": "LLM Bot WebSocket Server"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)