from typing import List, Optional, Literal, Annotated, TypedDict
from pydantic import (
    BaseModel,
    Field,
    ValidationError,
    model_validator,
    ConfigDict,
    field_validator
)
from datetime import datetime
from langchain_core.messages import (
    BaseMessage,
    AIMessage,
    HumanMessage,
    ToolMessage
)

from datetime import datetime

from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    requires_research: bool
    num_feedback_requests: int
    is_good_answer: bool
    messages: Annotated[List[BaseMessage], add_messages]
    research_cycles: int
    created_at: datetime
    last_updated: datetime

def validate_messages(messages: List[BaseMessage]) -> None:
    if not messages:
        raise ValueError("El estado debe contener al menos un mensaje")
    if not isinstance(messages[-1], (AIMessage, HumanMessage)):
        raise ValueError("Último mensaje debe ser del usuario o asistente")

def validate_state_consistency(state: AgentState) -> None:
    if state["is_good_answer"] and state["requires_research"]:
        raise ValueError("Estado inconsistente: respuesta buena pero requiere investigación")
    if state["num_feedback_requests"] > 0 and not state["requires_research"]:
        raise ValueError("Feedback solicitado sin investigación requerida")

class SearchPapersInput(BaseModel):
    """Input validado para búsquedas en CORE API"""
    query: str = Field(
        ...,
        min_length=3,
        max_length=200,
        description="Consulta de búsqueda para papers científicos",
    )
    
    max_papers: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Número máximo de papers a retornar",
    )

    @field_validator('query')
    @classmethod
    def sanitize_query(cls, value: str) -> str:
        """Sanitiza la consulta de búsqueda"""
        cleaned = value.strip().replace('"', '')
        if len(cleaned) < 3:
            raise ValueError("La consulta debe tener al menos 3 caracteres")
        return cleaned

class DecisionMakingOutput(BaseModel):
    """Output estructurado del nodo de toma de decisiones inicial"""
    requires_research: bool = Field(
        ...,
        description="Indica si la consulta requiere investigación adicional"
    )
    
    # Eliminamos min_length y max_length para evitar invalid schema en openai
    answer: Optional[str] = Field(
        default=None,
        description="Respuesta directa si no se requiere investigación"
    )

    @model_validator(mode='after')
    def validate_answer_presence(self) -> 'DecisionMakingOutput':
        if not self.requires_research and not self.answer:
            raise ValueError("Se requiere una respuesta cuando no hay necesidad de investigación")
        return self

class JudgeOutput(BaseModel):
    """Output estructurado del nodo de evaluación de calidad"""
    is_good_answer: bool = Field(
        ...,
        description="Indica si la respuesta cumple con los criterios de calidad"
    )
    
    # También quitamos min_length, max_length de 'feedback' si causara problemas
    feedback: Optional[str] = Field(
        default=None,
        description="Feedback detallado para mejorar la respuesta",
    )

    @model_validator(mode='after')
    def validate_feedback_presence(self) -> 'JudgeOutput':
        if not self.is_good_answer and not self.feedback:
            raise ValueError("Se requiere feedback cuando la respuesta no es satisfactoria")
        return self

SearchStep = Literal["decision_making", "planning", "tools", "agent", "judge"]
ValidationResult = Annotated[dict, Field(description="Resultado de validación del estado")]
