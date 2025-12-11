# 🧠 **LangGraph Multi-Agent Research Workflow**
### 📘 Automating Research, Analysis, and Reporting with Google Gemini (Gemini 2.0 Flash)

This tutorial demonstrates how to create an **autonomous multi-agent system** using **LangGraph** and **LangChain with Google Gemini**.  
Each agent performs a distinct function in a sequential workflow:
1. **Research Agent** → Gathers and structures data  
2. **Analysis Agent** → Extracts insights and patterns  
3. **Report Agent** → Generates a final professional report  

---

## ⚙️ Installation

Install required packages using pip:

```bash
!pip install -q langgraph langchain-google-genai langchain-core
```

---

## 🔧 Code Implementation

```python
# ---------------------------------------------
# IMPORTS AND SETUP
# ---------------------------------------------
import os
from typing import TypedDict, Annotated, List, Dict, Any
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
import operator
import json

# Set your Google API key
os.environ["GOOGLE_API_KEY"] = "YOUR_API_KEY"
```

---

## 🧩 Defining Agent State

The `AgentState` defines how information is passed between agents.

```python
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]  # Conversation history
    current_agent: str                                   # Active agent name
    research_data: dict                                  # Collected research data
    analysis_complete: bool                              # Flag after analysis
    final_report: str                                    # Final generated report
```

---

## 🤖 Initialize the LLM

Using **Gemini 2.0 Flash** (Google’s lightweight model):

```python
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7)
```

---

## 🌐 Simulated Tools

In a real application, these can be replaced with APIs (e.g., SerpAPI, Tavily, Firecrawl, etc.)

```python
def simulate_web_search(query: str) -> str:
    return f"Search results for '{query}': Found relevant information about {query} including recent developments, expert opinions, and statistical data."

def simulate_data_analysis(data: str) -> str:
    return f"Analysis complete: Key insights from the data include emerging trends, statistical patterns, and actionable recommendations."
```

---

## 🧠 Research Agent

Collects structured insights from simulated web searches.

```python
def research_agent(state: AgentState) -> AgentState:
    messages = state["messages"]
    last_message = messages[-1].content

    search_results = simulate_web_search(last_message)

    prompt = f"You are a research agent. Based on the query: '{last_message}'\nHere are the search results: {search_results}\nConduct thorough research and gather relevant information."

    response = llm.invoke([HumanMessage(content=prompt)])

    research_data = {
        "topic": last_message,
        "findings": response.content,
        "search_results": search_results,
        "sources": ["academic_papers", "industry_reports", "expert_analyses"],
        "confidence": 0.88,
        "timestamp": "2024-research-session"
    }

    return {
        "messages": state["messages"] + [AIMessage(content=f"Research completed on '{last_message}': {response.content}")],
        "current_agent": "analysis",
        "research_data": research_data,
        "analysis_complete": False,
        "final_report": ""
    }
```

---

## 📊 Analysis Agent

Analyzes research findings to produce insights and implications.

```python
def analysis_agent(state: AgentState) -> AgentState:
    research_data = state["research_data"]
    analysis_results = simulate_data_analysis(research_data.get('findings', ''))

    prompt = f"You are an analysis agent. Analyze this data deeply and provide insights for topic: {research_data.get('topic', 'Unknown')}."
    response = llm.invoke([HumanMessage(content=prompt)])

    return {
        "messages": state["messages"] + [AIMessage(content=f"Analysis completed: {response.content}")],
        "current_agent": "report",
        "research_data": state["research_data"],
        "analysis_complete": True,
        "final_report": ""
    }
```

---

## 🧾 Report Agent

Generates a **final structured research report** using the analysis and findings.

```python
def report_agent(state: AgentState) -> AgentState:
    research_data = state["research_data"]
    analysis_message = None

    for msg in reversed(state["messages"]):
        if isinstance(msg, AIMessage) and "Analysis completed:" in msg.content:
            analysis_message = msg.content.replace("Analysis completed: ", "")
            break

    prompt = f"Generate a comprehensive executive report based on research topic: {research_data.get('topic')} and analysis findings."

    response = llm.invoke([HumanMessage(content=prompt)])

    return {
        "messages": state["messages"] + [AIMessage(content=f'📄 FINAL REPORT GENERATED:\n\n{response.content}')],
        "current_agent": "complete",
        "research_data": state["research_data"],
        "analysis_complete": True,
        "final_report": response.content
    }
```

---

## 🔁 Workflow Logic

Defines the sequence: **Research → Analysis → Report → End**

```python
def should_continue(state: AgentState) -> str:
    current_agent = state.get("current_agent", "research")
    if current_agent == "research":
        return "analysis"
    elif current_agent == "analysis":
        return "report"
    elif current_agent == "report":
        return END
    else:
        return END
```

---

## 🧩 LangGraph Workflow Configuration

Building the full state graph of agents.

```python
workflow = StateGraph(AgentState)

workflow.add_node("research", research_agent)
workflow.add_node("analysis", analysis_agent)
workflow.add_node("report", report_agent)

workflow.add_conditional_edges("research", should_continue, {"analysis": "analysis", END: END})
workflow.add_conditional_edges("analysis", should_continue, {"report": "report", END: END})
workflow.add_conditional_edges("report", should_continue, {END: END})

workflow.set_entry_point("research")
app = workflow.compile()
```

---

## 🧭 Run the Multi-Agent Workflow

```python
def run_research_assistant(query: str):
    initial_state = {
        "messages": [HumanMessage(content=query)],
        "current_agent": "research",
        "research_data": {},
        "analysis_complete": False,
        "final_report": ""
    }

    print(f"🔍 Starting Multi-Agent Research on: '{query}'")
    print("=" * 60)

    current_state = research_agent(initial_state)
    print("✅ Research phase completed!")

    current_state = analysis_agent(current_state)
    print("✅ Analysis phase completed!")

    final_state = report_agent(current_state)
    print("✅ Report generation completed!")

    print("=" * 60)
    print("🎯 MULTI-AGENT WORKFLOW COMPLETED SUCCESSFULLY!")
    print("=" * 60)

    print(f"📋 COMPREHENSIVE RESEARCH REPORT:\n{final_state['final_report']}")
    return final_state
```

---

## ▶️ Example Usage

```python
if __name__ == "__main__":
    print("🚀 Running the LangGraph Multi-Agent Workflow!")
    result = run_research_assistant("What are emerging trends in sustainable technology?")
```

---

## 🌟 Key Learnings

| Concept | Description |
|----------|--------------|
| 🧠 Multi-Agent Design | Modular agents communicating via shared state |
| 🧩 LangGraph | State-based workflow engine for agent chaining |
| 🔍 Research Agent | Collects structured data |
| 📊 Analysis Agent | Derives insights and implications |
| 🧾 Report Agent | Produces final professional output |
| ☁️ Extensible | Replace simulated tools with real APIs or vector search |

---
