# LlmBot Crew

Welcome to the LlmBot Crew project, powered by [crewAI](https://crewai.com). This project implements a sophisticated multi-agent AI system that can process natural language commands, handle visual inputs, and provide real-time responses through multiple communication interfaces.

## Features

### Core Capabilities
- **Multi-Command Processing**: Intelligently parses and processes multiple commands from a single input
- **Standardized Measurements**: Automatically converts various units to standard measurements (cm, degrees)
- **Vision Analysis**: Processes visual data and provides natural language descriptions
- **Interactive Chat**: Handles conversational queries with context-aware responses
- **Flexible Deployment**: Supports both WebSocket and ZeroMQ communication protocols

### Agent System
Our system employs five specialized agents:
1. **Manager Agent**: Orchestrates command processing and ensures complete execution
2. **Command Processor**: Parses natural language into structured commands with standardized measurements
3. **Vision Agent**: Analyzes visual inputs and generates descriptive responses
4. **Chat Agent**: Handles conversational interactions with natural, context-aware responses
5. **Response Generator**: Formats outputs into standardized JSON responses

### Communication Interfaces
- **WebSocket Server** (`app.py`):
  - Real-time bidirectional communication
  - Supports base64 encoded image processing
  - JSON-based message format
  - CORS-enabled for web clients

- **ZeroMQ Server** (`app_zmq.py`):
  - High-performance message queuing
  - Request-Reply pattern implementation
  - Robust error handling
  - Graceful shutdown support

## Installation

Ensure you have Python >=3.10 <3.13 installed. This project uses [UV](https://docs.astral.sh/uv/) for dependency management.

```bash
# Install UV
pip install uv

# Install project dependencies
crewai install
```

### Configuration
1. Add your `OPENAI_API_KEY` to the `.env` file
2. Customize agents in `src/llm_bot/config/agents.yaml`
3. Modify tasks in `src/llm_bot/config/tasks.yaml`
4. Adjust core logic in `src/llm_bot/crew.py`

## Deployment

### WebSocket Server
```bash
# Start the WebSocket server
python app.py
```
The server will start on port 8000 by default.

### ZeroMQ Server
```bash
# Start the ZeroMQ server
python app_zmq.py
```
The server will start on port 5555 by default.

## Usage Examples

### WebSocket Client
```javascript
const ws = new WebSocket('ws://localhost:8000/ws');

ws.send(JSON.stringify({
    user_command: "Move forward 5 feet and rotate clockwise 90 degrees",
    image: "base64EncodedImageString" // optional
}));

ws.onmessage = (event) => {
    const response = JSON.parse(event.data);
    console.log(response);
};
```

### ZeroMQ Client
```python
from llm_bot.client import LLMBotClient

client = LLMBotClient()
response = client.send_command(
    "Move forward 5 feet and rotate clockwise 90 degrees",
    image_path="optional/path/to/image.jpg"
)
```

## Upcoming Features

### Vision Enhancements
- Integration of [Kosmos-2 VLM](https://huggingface.co/docs/transformers/en/model_doc/kosmos-2) (Visual Language Model)
- Improved scene understanding and object detection
- Enhanced spatial relationship analysis
- Real-time video stream processing support

### Additional Tools
- **Weather Integration**: Real-time weather data access
- **Location Services**: Geolocation and mapping capabilities
- **Time-Aware Responses**: Timezone and scheduling support
- **External Data Sources**: Integration with various APIs for real-time information

### System Improvements
- Distributed agent processing
- Enhanced error recovery
- Performance optimization
- Extended unit conversion support
- Improved natural language understanding

## Contributing
Contributions are welcomed!
