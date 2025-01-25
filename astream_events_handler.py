# astream_events_handler.py

import asyncio
import time
import logging
from typing import Dict, Any
import streamlit as st

from langchain_core.messages import BaseMessage
from graph import app_runnable

logger = logging.getLogger(__name__)

class ResearchSupervisor:
    """Clase mejorada para gestionar el ciclo de investigación."""
    def __init__(self):
        if "research_data" not in st.session_state:
            st.session_state.research_data = {
                "executions": [],
                "current_cycle": 0,
                "max_cycles": 10
            }
        
        self.tools: Dict[str, Dict[str, Any]] = {}
        self._cancelled = False

    def cancel_research(self):
        """Maneja la cancelación limpia de la investigación."""
        self._cancelled = True
        logger.info("Investigación cancelada por el usuario")
        st.toast("🛑 Investigación detenida", icon="⏹️")

    def update_progress(self, progress: float, message: str):
        """Actualiza la barra de progreso de forma segura."""
        try:
            if 'progress_bar' in st.session_state:
                st.session_state.progress_bar.progress(
                    min(progress, 1.0), 
                    text=message
                )
        except Exception as e:
            logger.error(f"Error actualizando progreso: {str(e)}")


async def execute_research_flow(
    messages: list,
    placeholder: st.delta_generator.DeltaGenerator
) -> str:
    """
    Flujo principal de investigación con manejo robusto de errores.
    """
    supervisor = ResearchSupervisor()
    final_text = ""
    
    try:
        # Inicialización del estado de herramientas
        if "tool_executions" not in st.session_state:
            st.session_state.tool_executions = []
        
        # Inicializa la barra de progreso
        if "progress_bar" not in st.session_state:
            st.session_state.progress_bar = placeholder.progress(0, text="🚀 Iniciando investigación...")
            st.session_state.current_progress = 0.0

        # Inicia el stream asincrónico del grafo
        research_stream = app_runnable.astream_events(
            {"messages": messages}, 
            version="v2",
            config={"recursion_limit": 30}  # Aumentamos el límite de recursión
        )
        
        async for event in research_stream:
            if supervisor._cancelled:
                raise asyncio.CancelledError()

            event_type = event["event"]

            # Manejo de ciclos de investigación
            if event_type == "on_chain_start" and event["name"] == "planning":
                st.session_state.research_data["current_cycle"] += 1
                logger.debug(f"Ciclo de investigación #{st.session_state.research_data['current_cycle']}")
                
                if st.session_state.research_data["current_cycle"] > st.session_state.research_data["max_cycles"]:
                    raise RuntimeError("🔬 Límite máximo de ciclos alcanzado. Revisa los parámetros de búsqueda.")
            
            # Actualización de progreso dinámico
            progress_weights = {
                "on_tool_start": 0.1,
                "on_tool_end": 0.2,
                "on_chat_model_stream": 0.05
            }
            st.session_state.current_progress = min(
                st.session_state.current_progress + progress_weights.get(event_type, 0),
                0.95
            )
            
            supervisor.update_progress(
                st.session_state.current_progress,
                "🔍 Analizando información..."
            )

            # Procesamiento de eventos relevantes
            if event_type == "on_chat_model_stream":
                final_text += event["data"]["chunk"].content
                placeholder.markdown(f"```markdown\n{final_text}\n```")

            elif event_type == "on_tool_start":
                # Registro persistente de la herramienta
                tool_data = {
                    "id": event["run_id"][:8],
                    "name": event["name"],
                    "input": event["data"].get("input", {}),
                    "start_time": time.time(),
                    "output": None,
                    "status": "running",
                    "execution_time": None
                }
                
                st.session_state.tool_executions.append(tool_data)
                st.toast(f"Iniciando: {tool_data['name']}", icon="⚡")

            elif event_type == "on_tool_end":
                # Actualización del estado de la herramienta
                output = event["data"].get("output", {})
                error = event["data"].get("error")
                
                if st.session_state.tool_executions:
                    last_tool = st.session_state.tool_executions[-1]
                    last_tool["output"] = output
                    last_tool["status"] = "error" if error else "success"
                    last_tool["execution_time"] = time.time() - last_tool["start_time"]
                
                with placeholder.container():
                    if error:
                        st.error(f"❌ Error en herramienta:\n```\n{str(error)[:500]}\n```")
                    else:
                        st.success("✅ Resultado obtenido")
                        st.json(output, expanded=False)

    except asyncio.CancelledError:
        logger.warning("Investigación cancelada por el usuario")
        placeholder.warning("⏹️ Investigación detenida a petición del usuario")
        return "Operación cancelada"
    
    except RuntimeError as e:
        logger.error(f"Límite de ciclos alcanzado: {str(e)}")
        placeholder.error(f"⚠️ {str(e)}")
        return "Límite máximo de iteraciones alcanzado"
    
    except Exception as e:
        logger.error(f"Error crítico: {str(e)}", exc_info=True)
        placeholder.error(f"🚨 Error inesperado: {str(e)}")
        return "Error en el proceso"
    
    finally:
        try:
            supervisor.update_progress(1.0, "🏁 Proceso completado")
            await asyncio.sleep(0.5)
            if "progress_bar" in st.session_state:
                st.session_state.progress_bar.empty()
                del st.session_state.progress_bar
            
            st.session_state.current_progress = 0.0
        except Exception as e:
            logger.error(f"Error en limpieza: {str(e)}")

    return final_text