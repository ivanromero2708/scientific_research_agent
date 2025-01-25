from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage, AIMessage
from langchain_core.runnables import RunnableConfig, Runnable
from pydantic import ValidationError
import logging
import os
import json
from typing import Dict, Any

from state import AgentState, DecisionMakingOutput, JudgeOutput
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
    """Clase para manejar la configuración del grafo"""
    def __init__(self):
        self.llm_model = "gpt-4-turbo"
        self.llm_temperature = 0.1
        self.max_research_cycles = 3
        self.max_feedback_attempts = 2
        
        self._validate_environment()

    def _validate_environment(self):
        """Valida las variables de entorno requeridas"""
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY no está configurada")
        if not os.getenv("CORE_API_KEY"):
            logger.warning("CORE_API_KEY no está configurada - algunas herramientas no funcionarán")

    def initialize_llms(self):
        """Inicializa los modelos de lenguaje con configuración validada"""
        return ChatOpenAI(
            model=self.llm_model,
            temperature=self.llm_temperature
        )

class EnhancedStateGraph(StateGraph):
    """Extensión mejorada de StateGraph con manejo de errores"""
    def __init__(self, state_schema):
        super().__init__(state_schema)
        self.error_state = None

    def add_error_handler(self, node_name: str):
        """Configura un manejador global de errores"""
        self.error_state = node_name
        return self

def setup_decision_making_node(config: GraphConfiguration):
    """Nodo de toma de decisiones inicial"""
    llm = config.initialize_llms()
    
    def _decision_making_node(state: AgentState) -> Dict[str, Any]:
        try:
            system_msg = SystemMessage(content=decision_making_prompt)
            response = llm.with_structured_output(DecisionMakingOutput).invoke(  # Usar la clase directamente
                [system_msg] + state.messages
            )
            
            logger.info(f"Decisión tomada: {'investigar' if response.requires_research else 'responder directamente'}")
            
            return {
                "requires_research": response.requires_research,
                "messages": [AIMessage(content=response.answer)] if response.answer else []
            }
            
        except ValidationError as e:
            logger.error(f"Error en formato de decisión: {str(e)}")
            return {"requires_research": True, "messages": []}

    return _decision_making_node

def setup_planning_node(config: GraphConfiguration):
    """Nodo de planificación de investigación"""
    llm = config.initialize_llms()
    tools_desc = format_tools_description(tools)
    
    def _planning_node(state: AgentState) -> Dict[str, Any]:
        try:
            enhanced_prompt = planning_prompt.format(
                tools=tools_desc,
                max_steps=config.max_research_cycles
            )
            
            system_msg = SystemMessage(content=enhanced_prompt)
            response = llm.invoke([system_msg] + state.messages)
            
            logger.info("Plan generado", extra={"plan": response.content})
            
            return {"messages": [response]}
            
        except Exception as e:
            logger.error(f"Error en planificación: {str(e)}")
            return {"messages": [AIMessage(content="Error en planificación. ¿Podrías reformular tu pregunta?")]}

    return _planning_node

def setup_tools_node():
    """Nodo de ejecución de herramientas"""
    tools_map = {tool.name: tool for tool in tools}
    
    def _tools_node(state: AgentState) -> Dict[str, Any]:
        outputs = []
        last_msg = state.messages[-1]
        
        if not last_msg.tool_calls:
            return {"messages": []}

        for tool_call in last_msg.tool_calls:
            try:
                tool = tools_map.get(tool_call["name"])
                if not tool:
                    raise KeyError(f"Herramienta {tool_call['name']} no encontrada")
                
                result = tool.invoke(tool_call["args"])
                outputs.append(
                    ToolMessage(
                        content=json.dumps(result),
                        name=tool_call["name"],
                        tool_call_id=tool_call["id"],
                    )
                )
                
                logger.info("Ejecución de herramienta exitosa", extra={
                    "tool": tool_call["name"],
                    "input": tool_call["args"]
                })
                
            except Exception as e:
                error_msg = f"Error en {tool_call['name']}: {str(e)}"
                outputs.append(
                    ToolMessage(
                        content=json.dumps({"error": error_msg}),
                        name=tool_call["name"],
                        tool_call_id=tool_call["id"],
                    )
                )
                logger.error(error_msg, exc_info=True)

        return {"messages": outputs}
    
    return _tools_node

def setup_agent_node(config: GraphConfiguration):
    """Nodo principal del agente"""
    llm = config.initialize_llms().bind_tools(tools)
    
    def _agent_node(state: AgentState) -> Dict[str, Any]:
        try:
            system_msg = SystemMessage(content=agent_prompt)
            config = RunnableConfig(metadata={
                "user_query": state.messages[-1].content,
                "research_cycle": state.num_feedback_requests + 1
            })
            
            response = llm.invoke([system_msg] + state.messages, config=config)
            
            logger.debug("Respuesta del agente generada", extra={
                "content": response.content[:100] + "..." if response.content else ""
            })
            
            return {"messages": [response]}
            
        except Exception as e:
            logger.error(f"Error en el nodo del agente: {str(e)}")
            return {"messages": [AIMessage(content="Ocurrió un error interno. ¿Podrías intentarlo de nuevo?")]}

    return _agent_node

def setup_judge_node(config: GraphConfiguration):
    """Nodo de evaluación de calidad"""
    llm = config.initialize_llms()
    
    def _judge_node(state: AgentState) -> Dict[str, Any]:
        if state.num_feedback_requests >= config.max_feedback_attempts:
            logger.warning("Límite de feedback alcanzado", extra={
                "attempts": state.num_feedback_requests
            })
            return {"is_good_answer": True}

        try:
            system_msg = SystemMessage(content=judge_prompt)
            response = llm.with_structured_output(AgentState.JudgeOutput).invoke(
                [system_msg] + state.messages
            )
            
            logger.info("Evaluación de calidad", extra={
                "es_buena_respuesta": response.is_good_answer,
                "intentos": state.num_feedback_requests
            })
            
            output = {
                "is_good_answer": response.is_good_answer,
                "num_feedback_requests": state.num_feedback_requests + 1
            }
            
            if response.feedback:
                output["messages"] = [AIMessage(content=response.feedback)]
                
            return output
            
        except ValidationError as e:
            logger.error(f"Error en evaluación: {str(e)}")
            return {"is_good_answer": True}

    return _judge_node

def create_workflow() -> Runnable:
    """Factory para crear el flujo de trabajo completo"""
    config = GraphConfiguration()
    
    workflow = EnhancedStateGraph(AgentState)
    
    # Nodos principales
    workflow.add_node("decision_making", setup_decision_making_node(config))
    workflow.add_node("planning", setup_planning_node(config))
    workflow.add_node("tools", setup_tools_node())
    workflow.add_node("agent", setup_agent_node(config))
    workflow.add_node("judge", setup_judge_node(config))

    # Transiciones
    workflow.set_entry_point("decision_making")
    
    workflow.add_conditional_edges(
        "decision_making",
        lambda state: "planning" if state["requires_research"] else END,
    )
    
    workflow.add_edge("planning", "agent")
    workflow.add_edge("tools", "agent")
    
    workflow.add_conditional_edges(
        "agent",
        lambda state: "tools" if state["messages"][-1].tool_calls else "judge",
    )
    
    workflow.add_conditional_edges(
        "judge",
        lambda state: "planning" if not state["is_good_answer"] else END,
    )
    
    # Manejador de errores global
    workflow.add_error_handler("planning")
    
    return workflow.compile()

# Inicialización final del grafo
app_runnable = create_workflow()