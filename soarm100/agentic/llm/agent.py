from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.graph import CompiledGraph
from langchain_core.messages import AIMessageChunk

from soarm100.agentic.robot import SOARM100AgenticPolicy
from soarm100.agentic.llm.prompts import SOARM100_PROMPT
from soarm100.agentic.llm.tools import (PutMarkerInBoxTool,
                                        StopRobotTool,
                                        TeleopRobotTool)

class AgentSOARM100:
    
    def __init__(self,robot_policy:SOARM100AgenticPolicy = None):
        
        self.llm = ChatOpenAI(model="gpt-4o-mini",temperature=0.5,streaming=True)
        
        self.robot_policy = robot_policy
        
        self.agent: CompiledGraph = create_react_agent(
            model=self.llm,
            tools=[PutMarkerInBoxTool(self.robot_policy),
                   StopRobotTool(self.robot_policy),
                   TeleopRobotTool(self.robot_policy)],
            checkpointer=MemorySaver(),
            state_modifier=SOARM100_PROMPT
        )

    def run(self,message:str):
        config = {"configurable": {"thread_id": "1"},"recursion_limit": 100}

        for event in self.agent.stream(input={"messages": [("user", message)]},
                                       config=config,
                                       stream_mode="messages"):
            message = event[0]
            if isinstance(message, AIMessageChunk):
                yield message.content
    

if __name__ == "__main__":
    agent = AgentSOARM100()
    for chunk in agent.run("Hola"):
        print(chunk)