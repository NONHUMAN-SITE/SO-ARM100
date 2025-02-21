from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.graph import CompiledGraph
from langchain.tools import tool
from soarm100.agentic.robot import SOARM100AgenticPolicy
from soarm100.agentic.llm.prompts import SOARM100_PROMPT
from soarm100.agentic.llm.tools import PutMarkerInBoxTool, StopRobotTool


class AgentSOARM100:
    
    def __init__(self,robot_policy:SOARM100AgenticPolicy):
        
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)
        
        self.robot_policy = robot_policy
        
        self.agent: CompiledGraph = create_react_agent(
            model=self.llm,
            tools=[PutMarkerInBoxTool(self.robot_policy),
                   StopRobotTool(self.robot_policy)],
            checkpointer=MemorySaver(),
            state_modifier=SOARM100_PROMPT
        )
        
    def transcribe(self, input_path:str) -> str:
        return self.stt.transcribe(input_path)

    def run(self,message:str) -> str:
        
        config = {
            "configurable": {"thread_id": "1"},
            "recursion_limit": 100,
        }
        
        state = self.agent.invoke({"messages": [("user", message)]},config)

        for message in state["messages"]:
            print(message)

        return state["messages"][-1].content