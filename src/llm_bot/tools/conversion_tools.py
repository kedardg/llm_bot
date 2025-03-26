from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field
import uuid
import json

class DistanceConversionInput(BaseModel):
    """Input schema for distance conversion tool."""
    value: float = Field(..., description="The value to convert.")
    unit: str = Field(..., description="The unit to convert from (feet, inches, meters, yards, etc.).")

class AngleConversionInput(BaseModel):
    """Input schema for angle conversion tool."""
    value: float = Field(..., description="The value to convert.")
    unit: str = Field(..., description="The unit to convert from (radians, mils, gradians, etc.).")

class DistanceConversionTool(BaseTool):
    name: str = "Distance Conversion Tool"
    description: str = (
        "Converts various distance units to centimeters (cm). "
        "Supported units: feet, inches, meters, yards, etc."
    )
    args_schema: Type[BaseModel] = DistanceConversionInput

    def _run(self, value: float, unit: str) -> float:
        unit = unit.lower().strip()
        # Convert to cm
        if unit in ["feet", "foot", "ft"]:
            return value * 30.48
        elif unit in ["inches", "inch", "in"]:
            return value * 2.54
        elif unit in ["meters", "meter", "m"]:
            return value * 100
        elif unit in ["yards", "yard", "yd"]:
            return value * 91.44
        elif unit in ["centimeters", "centimeter", "cm"]:
            return value
        elif unit in ["millimeters", "millimeter", "mm"]:
            return value * 0.1
        else:
            raise ValueError(f"Unsupported unit: {unit}")

class AngleConversionTool(BaseTool):
    name: str = "Angle Conversion Tool"
    description: str = (
        "Converts various angle units to degrees. "
        "Supported units: radians, mils, gradians, etc."
    )
    args_schema: Type[BaseModel] = AngleConversionInput

    def _run(self, value: float, unit: str) -> float:
        unit = unit.lower().strip()
        # Convert to degrees
        if unit in ["radians", "radian", "rad"]:
            import math
            return value * 180 / math.pi
        elif unit in ["mils", "mil"]:
            return value * 0.05625
        elif unit in ["gradians", "gradian", "grad"]:
            return value * 0.9
        elif unit in ["degrees", "degree", "deg"]:
            return value
        else:
            raise ValueError(f"Unsupported unit: {unit}")

class VisionInput(BaseModel):
    """Input schema for vision tool."""
    query: str = Field(..., description="The query about what to look for in the visual data.")

class VisionTool(BaseTool):
    name: str = "Vision Analysis Tool"
    description: str = (
        "Analyzes visual data and provides descriptions of what is seen. "
        "Use this tool when commands like 'tell me what you see' are given."
    )
    args_schema: Type[BaseModel] = VisionInput

    def _run(self, query: str) -> str:
        # In a real implementation, this would connect to a vision system
        # Here we're mocking the response to match the expected format
        
        # Generate mock vision response
        response = {
            "id": str(uuid.uuid4()),
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": {
                        "message": "A young family is sitting in the grass with their dog.",
                        "boundingBoxes": [{
                            "phrase": "A young family",
                            "substring": [0, 13],
                            "bboxes": [[0.046875, 0.015625, 0.453125, 0.984375], 
                                      [0.484375, 0.015625, 0.984375, 0.984375]]
                        }, {
                            "phrase": "their dog",
                            "substring": [44, 52],
                            "bboxes": [[0.390625, 0.578125, 0.640625, 0.984375]]
                        }]
                    },
                    "entities": [{
                        "phrase": "a multiplayer online game",
                        "substring": [12, 36],
                        "bboxes": [[0.078125, 0.046875, 0.921875, 0.234375]]
                    }]
                },
                "finish_reason": "stop"
            }]
        }
        
        # Return the full JSON response
        return json.dumps(response)

class ChatInput(BaseModel):
    """Input schema for chat tool."""
    message: str = Field(..., description="The message to process.")

class ChatTool(BaseTool):
    name: str = "Conversational Processing Tool"
    description: str = (
        "Processes conversation elements and generates natural language responses."
    )
    args_schema: Type[BaseModel] = ChatInput

    def _run(self, message: str) -> str:
        # In a real implementation, this would connect to a conversational AI
        # Here we're mocking a simple response
        return "A young family is sitting in the grass with their dog." 