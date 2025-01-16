from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langchain_core.messages import BaseMessage, AnyMessage, SystemMessage, ToolMessage, AIMessage
from langchain_openai import ChatOpenAI

import json
from dotenv import load_dotenv
import os

from state import (
    AgentState,
    DecisionMakingOutput,
    JudgeOutput,
)

from agent_tools import (
    tools
)

from utils import (
    decision_making_prompt,
    planning_prompt,
    format_tools_description,
    agent_prompt,
    judge_prompt,
)

load_dotenv()

# You can set your own keys here
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["CORE_API_KEY"] = os.getenv("CORE_API_KEY")

# LLMs
base_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
decision_making_llm = base_llm.with_structured_output(DecisionMakingOutput)
agent_llm = base_llm.bind_tools(tools)
judge_llm = base_llm.with_structured_output(JudgeOutput)

# Tools dictionary
tools_dict = {tool.name: tool for tool in tools}

# Decision making node
def decision_making_node(state: AgentState):
    """Entry point of the workflow. Based on the user query, the model can either respond directly or perform a full research, routing the workflow to the planning node"""
    system_prompt = SystemMessage(content=decision_making_prompt)
    response: DecisionMakingOutput = decision_making_llm.invoke([system_prompt] + state["messages"])
    output = {"requires_research": response.requires_research}
    if response.answer:
        output["messages"] = [AIMessage(content=response.answer)]
    return output

# Task router function
def router(state: AgentState):
    """Router directing the user query to the appropriate branch of the workflow."""
    if state["requires_research"]:
        return "planning"
    else:
        return "end"

# Planning node
def planning_node(state: AgentState):
    """Planning node that creates a step by step plan to answer the user query."""
    system_prompt = SystemMessage(content=planning_prompt.format(tools=format_tools_description(tools)))
    response = base_llm.invoke([system_prompt] + state["messages"])
    return {"messages": [response]}

# Tool call node
def tools_node(state: AgentState):
    """Tool call node that executes the tools based on the plan."""
    outputs = []
    for tool_call in state["messages"][-1].tool_calls:
        tool_result = tools_dict[tool_call["name"]].invoke(tool_call["args"])
        outputs.append(
            ToolMessage(
                content=json.dumps(tool_result),
                name=tool_call["name"],
                tool_call_id=tool_call["id"],
            )
        )
    return {"messages": outputs}

# Agent call node
def agent_node(state: AgentState):
    """Agent call node that uses the LLM with tools to answer the user query."""
    system_prompt = SystemMessage(content=agent_prompt)
    response = agent_llm.invoke([system_prompt] + state["messages"])
    return {"messages": [response]}

# Should continue function
def should_continue(state: AgentState):
    """Check if the agent should continue or end."""
    messages = state["messages"]
    last_message = messages[-1]

    # End execution if there are no tool calls
    if last_message.tool_calls:
        return "continue"
    else:
        return "end"

# Judge node
def judge_node(state: AgentState):
    """Node to let the LLM judge the quality of its own final answer."""
    # End execution if the LLM failed to provide a good answer twice.
    num_feedback_requests = state.get("num_feedback_requests", 0)
    if num_feedback_requests >= 2:
        return {"is_good_answer": True}

    system_prompt = SystemMessage(content=judge_prompt)
    response: JudgeOutput = judge_llm.invoke([system_prompt] + state["messages"])
    output = {
        "is_good_answer": response.is_good_answer,
        "num_feedback_requests": num_feedback_requests + 1
    }
    if response.feedback:
        output["messages"] = [AIMessage(content=response.feedback)]
    return output

# Final answer router function
def final_answer_router(state: AgentState):
    """Router to end the workflow or improve the answer."""
    if state["is_good_answer"]:
        return "end"
    else:
        return "planning"

# Initialize the StateGraph
workflow = StateGraph(AgentState)

# Add nodes to the graph
workflow.add_node("decision_making", decision_making_node)
workflow.add_node("planning", planning_node)
workflow.add_node("tools", tools_node)
workflow.add_node("agent", agent_node)
workflow.add_node("judge", judge_node)

# Set the entry point of the graph
workflow.set_entry_point("decision_making")

# Add edges between nodes
workflow.add_conditional_edges(
    "decision_making",
    router,
    {
        "planning": "planning",
        "end": END,
    }
)
workflow.add_edge("planning", "agent")
workflow.add_edge("tools", "agent")
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "tools",
        "end": "judge",
    },
)
workflow.add_conditional_edges(
    "judge",
    final_answer_router,
    {
        "planning": "planning",
        "end": END,
    }
)

# Compile the graph
app_runnable = workflow.compile()