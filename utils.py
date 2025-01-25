# utils.py - Versión Final Corregida
import re
import json
import logging
import jsonschema
from datetime import datetime
from typing import List, Optional, Dict, Any
from langchain_core.tools import BaseTool
from langchain_core.messages import BaseMessage
from IPython.display import display, Markdown

logger = logging.getLogger(__name__)

# --------------------------
# PROMPT TEMPLATES (CORRECTED FORMAT)
# --------------------------

def current_date():
    return datetime.now().strftime("%Y-%m-%d")

decision_making_prompt = f"""**Role**: Senior Scientific Research Assistant
**Current Date**: {current_date()}

# Objective:
Determine if the user query requires:
1. Literature review
2. Document analysis
3. Human intervention

# Decision Criteria:
✅ **Direct Answer** if:
- General methodology questions
- Conceptual explanations
- Questions about system capabilities

✅ **Requires Research** if:
- Mentions specific papers/documents
- Requests recent data (last 5 years)
- Requires experimental validation

# Instructions:
1. Analyze the latest user question
2. Classify using criteria
3. When in doubt, prioritize research"""

planning_prompt = """**Role**: Scientific Research Planner

# Objective:
Create a {max_steps}-step plan using available tools.

# Available Tools:
{tools}

# Rules:
1. Prioritize recent sources (≥2020)
2. Cross-validate multiple sources
3. Limit searches to 3-5 key papers
4. Include validation steps

# Format:
```markdown
1. [Tool Name]: [Specific action]
   - Parameters: [key: value]
   
   Example:
   1. search-papers: Find recent CRISPR reviews
   - query: "CRISPR gene editing review"
   - max_papers: 3
2. download-paper: Get key paper
   - url: [First result URL]
```"""

agent_prompt = """**Role**: AI Research Scientist

# Guidelines:
1. Base all claims on peer-reviewed sources
2. Use [Author, Year] citation format
3. Highlight conflicting evidence
4. Differentiate consensus from hypotheses
5. Include DOI links when available

# Recommended Structure:
1. Executive Summary (50 words max)
2. Key Findings (bullet points)
3. Methodology Overview
4. Limitations & Biases
5. Future Research Directions

# Quality Control:
- Verify publication impact factor > 3.0
- Prefer open-access sources
- Check for retraction notices"""

judge_prompt = """**Role**: Scientific Quality Reviewer

# Evaluation Rubric:
| Criterion          | Weight | Description                  |
|--------------------|--------|------------------------------|
| Relevance          | 30%    | Addresses research question  |
| Rigor              | 25%    | Methodology quality           |
| Currency           | 20%    | ≤5 year sources              |
| Transparency       | 15%    | Disclosure of limitations    |
| Impact             | 10%    | Practical significance       |

# Rating Scale:
1. Reject - Major flaws
2. Major Revision Required
3. Minor Revision Needed
4. Accept with Suggestions
5. Publish as Is"""

# --------------------------
# TOOL FORMATTING (CORRECTED)
# --------------------------

def format_tools_description(tools: List[BaseTool]) -> str:
    """Generate professional documentation for research tools"""
    tool_docs = []
    for tool in tools:
        try:
            params = tool.args_schema.schema() if tool.args_schema else {}
            param_table = "\n".join(
                f"- **{name}**: {schema.get('description', '')} "
                f"(Type: {schema.get('type', 'str')}, "
                f"Example: {schema.get('example', 'N/A')})"
                for name, schema in params.get("properties", {}).items()
            )
            
            example_args = json.dumps(tool.example, indent=2) if hasattr(tool, 'example') else ""
            
            tool_doc = f"""
## 🛠️ {tool.name}
**{tool.description}**

### Parameters:
{param_table}

### Example Usage:
```python
from agent_tools import {tool.name.replace('-', '_')}

result = {tool.name}(
    {example_args}
)
```"""
            tool_docs.append(tool_doc)
        except Exception as e:
            logger.error(f"Error documenting tool {tool.name}: {str(e)}")
            continue
    
    return "\n\n---\n\n".join(tool_docs)

# --------------------------
# STREAM HANDLING (CORRECTED)
# --------------------------

async def print_stream(app, input: str) -> Optional[BaseMessage]:
    """Execute and display research workflow with scientific formatting"""
    session_header = f"### 🔬 Research Session - {datetime.now().strftime('%Y-%m-%d %H:%M')}\n**Query**: \"{input[:100]}{'...' if len(input) > 100 else ''}\""
    display(Markdown(session_header))
    
    messages = []
    tool_counter = 1
    
    try:
        async for event in app.astream({"messages": [input]}, stream_mode="updates"):
            for node, updates in event.items():
                current_messages = updates.get("messages", [])
                for msg in current_messages:
                    if msg not in messages:
                        messages.append(msg)
                        
                        if msg.type == "ai":
                            display(Markdown(f"**Analysis Preview**\n```markdown\n{_format_research_output(msg.content)}\n```"))
                        elif msg.type == "tool":
                            display(Markdown(f"🔍 **Tool #{tool_counter}**: {msg.name}\n```json\n{_format_tool_data(msg.content)}\n```"))
                            tool_counter += 1
    
    except Exception as e:
        error_msg = f"❗ **Research Interrupted**\n```error\n{str(e)}\n```"
        display(Markdown(error_msg))
        logger.error(f"Research workflow failed: {str(e)}")
        return None
    
    final_output = f"🎯 **Final Research Output**\n{_format_research_output(messages[-1].content)}"
    display(Markdown(final_output))
    
    return messages[-1] if messages else None

# --------------------------
# FORMATTING UTILITIES (CORRECTED)
# --------------------------

def _format_research_output(text: str) -> str:
    """Format scientific content with proper typography"""
    text = re.sub(r"(\d)\s([%‰°C])", r"\1\u00A0\2", text)  # Non-breaking spaces
    text = re.sub(r"(\d+)(\.)(\d+)", r"\1\2\3", text)      # Decimal alignment
    text = re.sub(r"\b(\d+)\s*-\s*(\d+)\b", r"\1–\2", text)  # En-dash
    
    # Highlight key terms
    keywords = {
        r"\bhypothesis\b": "**hypothesis**",
        r"\bmethodology\b": "**methodology**",
        r"\bp-value\b": "`p-value`",
        r"\bCI\b": "`CI`"
    }
    for pattern, replacement in keywords.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    
    return text

def _format_tool_data(data: str) -> str:
    """Structure tool output for scientific readability"""
    try:
        json_data = json.loads(data)
        formatted = json.dumps(json_data, indent=2)
        if len(formatted) > 2000:
            return formatted[:800] + "\n... [truncated] ...\n" + formatted[-800:]
        return formatted
    except json.JSONDecodeError:
        return data[:2000]

# --------------------------
# VALIDATION (CORRECTED)
# --------------------------

def validate_research_schema(data: Dict[str, Any]) -> bool:
    """Validate scientific data against ISO 2145 schema"""
    schema = {
        "type": "object",
        "properties": {
            "hypothesis": {"type": "string"},
            "methodology": {"type": "string"},
            "results": {
                "type": "object",
                "properties": {
                    "sample_size": {"type": "integer"},
                    "confidence_interval": {"type": "array", "items": {"type": "number"}},
                    "p_value": {"type": "number"}
                },
                "required": ["sample_size"]
            },
            "citations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "author": {"type": "string"},
                        "year": {"type": "integer"},
                        "doi": {"type": "string"}
                    }
                }
            }
        },
        "required": ["hypothesis", "methodology"]
    }
    
    try:
        jsonschema.validate(instance=data, schema=schema)
        return True
    except jsonschema.ValidationError as e:
        logger.error(f"Schema validation failed: {str(e)}")
        return False