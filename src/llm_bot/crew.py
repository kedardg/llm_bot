from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from llm_bot.tools.conversion_tools import (
    DistanceConversionTool, 
    AngleConversionTool,
    VisionTool,
    ChatTool
)
from pydantic import BaseModel, Field
from typing import Optional, Union, Literal, Dict, Any, List
import json
from pydantic import validator

# Add validation metadata models
class ValidationStatus(BaseModel):
    status: Literal["PASS", "FAIL"] = Field(..., description="Overall validation status")
    missing_commands: Optional[List[str]] = Field(None, description="Original text of any missing commands")
    validation_details: Optional[Dict[str, Any]] = Field(None, description="Additional validation information")

# Update CommandResponse to include processing metadata
class CommandResponse(BaseModel):
    command: Optional[Literal["MOVE_FORWARD", "MOVE_BACKWARD", "ROTATE_CLOCKWISE", "ROTATE_COUNTERCLOCKWISE", None]] = Field(
        None, 
        description="The command issued to the robot: MOVE_FORWARD, MOVE_BACKWARD, ROTATE_CLOCKWISE, ROTATE_COUNTERCLOCKWISE, or null if no command"
    )
    linear_distance: Optional[float] = Field(
        None, 
        description="Distance in centimeters for forward/backward movement, or null if not applicable"
    )
    rotate_degree: Optional[float] = Field(
        None, 
        description="Rotation in degrees for clockwise/counterclockwise rotation, or null if not applicable"
    )
    description: str = Field(
        ..., 
        description="Brief description of the scene and/or response to user query (1-2 sentences)"
    )
    raw_output: Optional[Dict[str, Any]] = Field(
        None,
        description="Raw unprocessed output from specialized tools/agents (vision, chat) for later reference"
    )
    processed_by: str = Field(
        ...,
        description="Agent that processed this command"
    )
    conversion_source: Optional[str] = Field(
        None,
        description="Source of any unit conversions (e.g., 'rule-based', 'distance_tool')"
    )

# Update BotResponseModel to include validation
class BotResponseModel(BaseModel):
    responses: List[CommandResponse] = Field(
        ...,
        description="List of all commands extracted from the user input, with each command properly processed"
    )
    validation: ValidationStatus = Field(
        ...,
        description="Validation status and details"
    )

    @validator('responses')
    def validate_responses_count(cls, v, values, **kwargs):
        """Ensure we have at least one response"""
        if not v:
            raise ValueError("At least one response is required")
        return v

@CrewBase
class LlmBot():
    """LlmBot crew for command processing and response generation"""

    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    # Define agents with their specific tools
    @agent
    def manager_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['manager_agent'],
            verbose=True,
            allow_delegation=True,
            backstory_additions="When delegating tasks, provide task descriptions as simple strings, not complex objects."
        )

    @agent
    def command_processor_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['command_processor_agent'],
            tools=[
                DistanceConversionTool(), 
                AngleConversionTool()
            ],
            verbose=True,
            allow_delegation=True,
            memory=False
        )

    @agent
    def vision_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['vision_agent'],
            tools=[
                VisionTool(result_as_answer=True)  # Force tool output as result
            ],
            verbose=True,
            force_tool_output=True,  # Ensure the tool output is returned directly
            allow_delegation=False
        )

    @agent
    def chat_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['chat_agent'],
            tools=[
                ChatTool(result_as_answer=True)  # Force tool output as result
            ],
            verbose=True,
            force_tool_output=True,  # Ensure the tool output is returned directly
            allow_delegation=False
        )

    @agent
    def response_generator_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['response_generator_agent'],
            verbose=True,
            allow_delegation=False
        )

    # Define tasks
    @task
    def command_processing_task(self) -> Task:
        return Task(
            config=self.tasks_config['command_processing_task'],
            expected_output="""
            {
                "responses": [
                    {
                        "command_type": "string",
                        "original_text": "string",
                        "value": "number or null",
                        "unit": "string or null"
                    }
                ]
            }
            """,
            max_retries=3
        )

    @task
    def unit_conversion_task(self) -> Task:
        return Task(
            config=self.tasks_config['unit_conversion_task'],
            context=[self.command_processing_task()],
            expected_output="""
            {
                "responses": [
                    {
                        "command_type": "string",
                        "original_text": "string",
                        "value": "number or null",
                        "unit": "string or null",
                        "converted_value": "number or null",
                        "converted_unit": "string or null"
                    }
                ]
            }
            """,
            max_retries=2
        )

    @task
    def vision_task(self) -> Task:
        return Task(
            config=self.tasks_config['vision_task'],
            context=[self.unit_conversion_task()],
            tools=[VisionTool(result_as_answer=True)],
            expected_output="""
            {
                "responses": [
                    {
                        "command": "string or null",
                        "vision_description": "string",
                        "raw_output": "object or null"
                    }
                ]
            }
            """,
            max_retries=2
        )

    @task
    def chat_task(self) -> Task:
        return Task(
            config=self.tasks_config['chat_task'],
            context=[self.unit_conversion_task()],
            tools=[ChatTool(result_as_answer=True)],
            expected_output="""
            {
                "responses": [
                    {
                        "command": "string or null",
                        "chat_response": "string",
                        "raw_output": "object or null"
                    }
                ]
            }
            """,
            max_retries=2
        )

    @task
    def response_generation_task(self) -> Task:
        # def validate_complete_responses(result):
        #     """Enhanced validation with detailed reporting"""
        #     try:
        #         # Handle different result types properly
        #         if hasattr(result, 'dict'):
        #             data = result.dict()
        #         elif hasattr(result, 'model_dump'):
        #             data = result.model_dump()
        #         elif isinstance(result, str):
        #             data = json.loads(result)
        #         else:
        #             data = result
                    
        #         # Force-fail if result is empty or invalid
        #         if not data:
        #             return False, "Empty or invalid result received"
                    
        #         # Access 'responses' with safer approach and detailed error handling
        #         responses = []
        #         if isinstance(data, dict) and 'responses' in data:
        #             responses = data['responses']
        #         elif hasattr(data, 'responses'):
        #             responses = data.responses
        #         else:
        #             return False, f"Invalid response format. Expected 'responses' array, got: {type(data)}"
                
        #         # Enhanced validation with detailed error collection
        #         validation_details = {
        #             "total_commands_processed": len(responses),
        #             "command_types": {},
        #             "missing_fields_by_response": {},
        #             "type_errors": [],
        #         }

        #         # Track command types for validation
        #         for resp in responses:
        #             cmd_type = resp.get('command') or resp.get('processed_by', 'unknown')
        #             validation_details["command_types"][cmd_type] = validation_details["command_types"].get(cmd_type, 0) + 1

        #         # Enhanced validation with detailed error collection
        #         missing_commands = []
        #         for i, resp in enumerate(responses):
        #             try:
        #                 resp_dict = (resp.dict() if hasattr(resp, 'dict') 
        #                            else resp.model_dump() if hasattr(resp, 'model_dump') 
        #                            else resp if isinstance(resp, dict) 
        #                            else {})
                        
        #                 # Validate required fields
        #                 missing_fields = []
        #                 invalid_fields = []
                        
        #                 for field, expected_type in [
        #                     ('command', (str, type(None))),
        #                     ('linear_distance', (float, int, type(None))),
        #                     ('rotate_degree', (float, int, type(None))),
        #                     ('description', str),
        #                     ('processed_by', str)
        #                 ]:
        #                     if field not in resp_dict:
        #                         missing_fields.append(field)
        #                     elif not isinstance(resp_dict[field], expected_type):
        #                         invalid_fields.append(f"{field} (expected {expected_type}, got {type(resp_dict[field])})")
                        
        #                 if missing_fields or invalid_fields:
        #                     validation_details["missing_fields_by_response"][i] = {
        #                         "missing_fields": missing_fields,
        #                         "invalid_fields": invalid_fields
        #                     }
                            
        #             except Exception as e:
        #                 validation_details["type_errors"].append(f"Response {i}: {str(e)}")

        #         # Determine validation status
        #         has_errors = (
        #             bool(validation_details["missing_fields_by_response"]) or 
        #             bool(validation_details["type_errors"]) or 
        #             len(responses) < 1
        #         )

        #         # Add validation status to the result
        #         if isinstance(data, dict):
        #             data["validation"] = {
        #                 "status": "FAIL" if has_errors else "PASS",
        #                 "missing_commands": missing_commands if missing_commands else None,
        #                 "validation_details": validation_details
        #             }

        #         return not has_errors, data
                
        #     except Exception as e:
        #         return False, f"Critical validation error: {str(e)}"
        
        model_info = {
            "responses": "Type: array. A list of command response objects, one for each command extracted from user input",
            "Each response contains": {
                "command": "Type: string or null. One of: 'MOVE_FORWARD', 'MOVE_BACKWARD', 'ROTATE_CLOCKWISE', 'ROTATE_COUNTERCLOCKWISE', or null",
                "linear_distance": "Type: number or null. Distance in centimeters (cm) for movement commands, or null if not applicable",
                "rotate_degree": "Type: number or null. Rotation angle in degrees for rotation commands, or null if not applicable",
                "description": "Type: string. Brief 1-2 sentence description of scene or response to user query",
                "raw_output": "Type: object or null. If vision or chat tools were used, include their complete raw output here; otherwise null",
                "processed_by": "Type: string. Agent that processed this command",
                "conversion_source": "Type: string or null. Source of any unit conversions (e.g., 'rule-based', 'distance_tool'), or null if not applicable"
            }
        }
        
        return Task(
            config=self.tasks_config['response_generation_task'],
            context=[self.unit_conversion_task(), self.vision_task(), self.chat_task()],
            output_pydantic=BotResponseModel,
            # guardrail=validate_complete_responses,
            max_retries=3,
            description_additions=f"""
            JSON OUTPUT STRUCTURE: {model_info}. 
            
            CRITICAL REQUIREMENTS:
            1. You MUST include ALL commands from the user input in your response
            2. Each command MUST be a separate entry in the responses array
            3. Vision or chat outputs MUST be preserved in raw_output
            4. ALL responses MUST follow the exact schema
            5. Minimum 3 responses are typically expected
            
            IMPORTANT FORMAT INSTRUCTIONS:
            1. When delegating work, provide task descriptions as simple strings
            2. Do not nest JSON objects when creating command responses
            3. Your final output must be valid JSON that matches the BotResponseModel schema exactly
            
            VALIDATION CHECKLIST:
            - Each response has all required fields
            - Commands are properly formatted
            - Numeric values are correct type
            - Raw outputs are preserved
            - All user commands are processed
            """,
            output_file='response.json'
        )

    @crew
    def crew(self) -> Crew:
        """Creates the command processing crew with hierarchical process"""
        try:
            manager_llm = LLM(model="gpt-4o")
        except Exception as e:
            print(f"Warning: Failed to create manager LLM with gpt-4o: {e}")
            manager_llm = None
        
        return Crew(
            agents=[
                self.command_processor_agent(),
                self.vision_agent(),
                self.chat_agent(),
                self.response_generator_agent()
            ],
            tasks=[
                self.command_processing_task(),
                self.unit_conversion_task(),
                self.vision_task(),
                self.chat_task(),
                self.response_generation_task()
            ],
            process=Process.hierarchical,
            manager_agent=self.manager_agent(),
            manager_llm=manager_llm,
            verbose=True
        )