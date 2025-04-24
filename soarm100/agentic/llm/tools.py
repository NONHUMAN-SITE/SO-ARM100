from typing import Literal, Optional

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool
from langchain_core.tools.base import ArgsSchema
from langchain_core.utils.function_calling import convert_to_openai_function
from pydantic import BaseModel, Field

from soarm100.agentic.llm.prompts import TASKS_ROBOT


def convert_to_tool_openai(tool):
    openai_tool = convert_to_openai_function(tool)
    openai_tool["type"] = "function"
    return openai_tool

class GiveInstructionSchema(BaseModel):
    instruction: TASKS_ROBOT = Field(
        description="The instruction to give to the physical robot. Must be one of the predefined instructions."
    )

class GiveInstructions(BaseTool):
    name: str = "give_instructions"
    description: str = "Use this tool to give instructions to the AI."
    args_schema: Optional[ArgsSchema] = GiveInstructionSchema
    return_direct: bool = True

    def _run(self, instructions: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        """Use the tool."""
        return f"Executing: {instructions}"
    
    async def _arun(self, instructions: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        """Use the tool asynchronously."""
        return f"Executing: {instructions}"

class GetFeedbackSchema(BaseModel):
    consult_query: str = Field(
        description=(
            "A question or query to check the current status or behavior of the robot. "
            "Useful for getting real-time feedback on the robot's ongoing task or operation. "
            "e.g. 'Has the robot finished cleaning?', 'What is the robot doing right now?', "
            "'Is the robot still navigating to the kitchen?'"
        )
    )
    task: str = Field(
        description=(
            "A brief description of the robot's current task or objective. "
            "e.g. 'Vacuuming the living room', 'Delivering a package to the second floor', "
            "'Inspecting shelves in aisle 3'"
        )
    )

class GetFeedback(BaseTool):
    name: str = "get_feedback"
    description: str = "Use this tool to get feedback from the Supervisor"
    args_schema: Optional[ArgsSchema] = GetFeedbackSchema
    return_direct: bool = True

    def _run(self, feedback: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        """Use the tool."""
        return f"Executing: {feedback}"
    
    async def _arun(self, feedback: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        """Use the tool asynchronously."""
        return f"Executing: {feedback}"

GIVE_INSTRUCTIONS_TOOL = convert_to_tool_openai(GiveInstructions())
GET_FEEDBACK_TOOL = convert_to_tool_openai(GetFeedback())
    