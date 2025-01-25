from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage, AIMessage
from langchain_core.runnables import RunnableConfig, Runnable
import logging
import os
import json
from typing import Dict, Any, Callable

from state import AgentState, DecisionMakingOutput, JudgeOutput, validate_messages, validate_state_consistency
from agent_tools import tools
from utils import (
    decision_making_prompt,
    planning_prompt,
    format_tools_description,
    agent_prompt,
    judge_prompt
)

logger = logging.getLogger(__name__)

class GraphConfiguration:
    """Manejador centralizado de configuración del grafo"""
    def __init__(self):
        self.llm_model = "gpt-4o-mini"
        self.llm_temperature = 0
        self.max_research_cycles = 3
        self.max_feedback_attempts = 2
        self._validate_environment()

    def _validate_environment(self):
        """Valida dependencias externas"""
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY no configurada")
        if not os.getenv("CORE_API_KEY"):
            logger.warning("CORE_API_KEY no encontrada - funcionalidad limitada")

    def initialize_llms(self) -> ChatOpenAI:
        """Factory para modelos de lenguaje"""
        return ChatOpenAI(
            model=self.llm_model,
            temperature=self.llm_temperature,
            #model_kwargs={"response_format": {"type": "json_object"}}
        )

def setup_decision_making_node(config: GraphConfiguration) -> Callable[[AgentState], Dict[str, Any]]:
    """Nodo de decisión inicial: Determina si se requiere investigación"""
    llm = config.initialize_llms()
    
    def _node_logic(state: AgentState) -> Dict[str, Any]:
        try:
            validate_messages(state["messages"])
            system_msg = SystemMessage(content=decision_making_prompt)
            response = llm.with_structured_output(DecisionMakingOutput).invoke(
                [system_msg] + state["messages"]
            )
            
            logger.info(f"Decisión: {'investigar' if response.requires_research else 'responder'}")
            
            return {
                "requires_research": response.requires_research,
                "messages": [AIMessage(content=response.answer)] if response.answer else []
            }
        except Exception as e:
            logger.error(f"Error en decisión: {str(e)}")
            return {"requires_research": True, "messages": []}

    return _node_logic

def setup_planning_node(config: GraphConfiguration) -> Callable[[AgentState], Dict[str, Any]]:
    """Nodo de planificación: Genera estrategia de investigación"""
    llm = config.initialize_llms()
    
    def _node_logic(state: AgentState) -> Dict[str, Any]:
        try:
            tools_desc = format_tools_description(tools)
            prompt = planning_prompt.format(tools=tools_desc, max_steps=config.max_research_cycles)
            system_msg = SystemMessage(content=prompt)
            response = llm.invoke([system_msg] + state["messages"])
            
            logger.info("Plan generado", extra={"plan": response.content})
            
            return {"messages": [response]}
        except Exception as e:
            logger.error(f"Error en planificación: {str(e)}")
            return {"messages": [AIMessage(content="Error en planificación. ¿Podrías reformular tu pregunta?")]}

    return _node_logic

def setup_tools_node() -> Callable[[AgentState], Dict[str, Any]]:
    """Ejecutor de herramientas: Maneja llamados a APIs externas"""
    tools_map = {tool.name: tool for tool in tools}
    
    def _node_logic(state: AgentState) -> Dict[str, Any]:
        try:
            last_msg = state["messages"][-1]
            outputs = []
            
            for tool_call in last_msg.tool_calls:
                tool = tools_map.get(tool_call["name"])
                if not tool:
                    raise KeyError(f"Herramienta {tool_call['name']} no registrada")
                
                result = tool.invoke(tool_call["args"])
                outputs.append(
                    ToolMessage(
                        content=json.dumps(result),
                        name=tool_call["name"],
                        tool_call_id=tool_call["id"],
                    )
                )
                logger.info("Ejecución exitosa", extra={"tool": tool_call["name"]})
            
            return {"messages": outputs}
        except Exception as e:
            logger.error(f"Error en herramientas: {str(e)}")
            return {"messages": [AIMessage(content=f"Error ejecutando herramienta: {str(e)}")]}

    return _node_logic

def setup_agent_node(config: GraphConfiguration) -> Callable[[AgentState], Dict[str, Any]]:
    """Nodo principal del agente: Genera respuestas usando LLM"""
    llm = config.initialize_llms().bind_tools(tools)
    
    def _node_logic(state: AgentState) -> Dict[str, Any]:
        try:
            system_msg = SystemMessage(content=agent_prompt)
            response = llm.invoke(
                [system_msg] + state["messages"],
                config=RunnableConfig(metadata={
                    "research_cycle": state.get("research_cycles", 0) + 1
                })
            )
            return {"messages": [response]}
        except Exception as e:
            logger.error(f"Error en agente: {str(e)}")
            return {"messages": [AIMessage(content="Error interno. Intenta nuevamente.")]}

    return _node_logic

def setup_judge_node(config: GraphConfiguration) -> Callable[[AgentState], Dict[str, Any]]:
    """Nodo de evaluación: Control de calidad de respuestas"""
    llm = config.initialize_llms()
    
    def _node_logic(state: AgentState) -> Dict[str, Any]:
        try:
            if state.get("num_feedback_requests", 0) >= config.max_feedback_attempts:
                logger.warning("Límite de feedback alcanzado")
                return {"is_good_answer": True}

            system_msg = SystemMessage(content=judge_prompt)
            response = llm.with_structured_output(JudgeOutput).invoke(
                [system_msg] + state["messages"]
            )
            
            output = {
                "is_good_answer": response.is_good_answer,
                "num_feedback_requests": state.get("num_feedback_requests", 0) + 1
            }
            
            if response.feedback:
                output["messages"] = [AIMessage(content=response.feedback)]
            
            validate_state_consistency(output)
            return output
        except Exception as e:
            logger.error(f"Error en evaluación: {str(e)}")
            return {"is_good_answer": True}

    return _node_logic

def create_workflow() -> Runnable:
    """Configuración final del grafo de flujo de trabajo"""
    config = GraphConfiguration()
    workflow = StateGraph(AgentState)

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
    
    # Task router function
    def router(state: AgentState):
        """Router directing the user query to the appropriate branch of the workflow."""
        if state["requires_research"]:
            return "planning"
        else:
            return "end"
    
    # Final answer router function
    def final_answer_router(state: AgentState):
        """Router to end the workflow or improve the answer."""
        if state["is_good_answer"]:
            return "end"
        else:
            return "planning"
    
    # Registro de nodos
    nodes = {
        "decision_making": setup_decision_making_node(config),
        "planning": setup_planning_node(config),
        "tools": setup_tools_node(),
        "agent": setup_agent_node(config),
        "judge": setup_judge_node(config)
    }
    
    for node_name, node in nodes.items():
        workflow.add_node(node_name, node)

    # Configuración de flujo
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

    return workflow.compile()

# Instancia ejecutable del grafo
app_runnable = create_workflow()