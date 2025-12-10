# This script demonstrates a baseline single-agent prototype using CrewAI
# with tool integration (Tavily Search) and the Groq LLM.
#
# Prerequisites:
# 1. Install necessary libraries: pip install crewai tavily-python 'crewai[tools]' litellm
# 2. Set environment variables for API keys:
#    - TAVILY_API_KEY: Required for the search tool.
#    - GROQ_API_KEY: Required for the Groq LLM.

import os
from crewai import Agent, Task, Crew, LLM
from crewai_tools import TavilySearchTool
from dotenv import load_dotenv

# Load environment variables (optional, but recommended for API keys)
load_dotenv()
    
# --- 1. CONFIGURATION CHECK & LLM SETUP ---
if not os.getenv("TAVILY_API_KEY") or not os.getenv("GROQ_API_KEY"):
    print("FATAL ERROR: Please set both TAVILY_API_KEY and GROQ_API_KEY environment variables.")
    # Exit cleanly if keys are missing to prevent runtime errors
    exit()

# Instantiate the Groq LLM
# The model 'llama-3.1-8b-instant' is used.
llm = LLM(model = "groq/llama-3.1-8b-instant", max_completion_tokens = 256)

# --- 2. TOOL DEFINITION ---
# Initialize the Tavily search tool. It automatically uses the TAVILY_API_KEY.
tavily_tool = TavilySearchTool()

# --- 3. AGENT DEFINITION (The Single Agent) ---
researcher_agent = Agent(
    role = 'Simple Fact Checker', # Agent Role
    goal = 'Quickly and accurately retrieve a single, recent factual data point using web search.', # Agent Goal
    backstory = (
        "A reliable data retrieval bot known for its efficiency and speed in answering specific, single-point queries. "
        "Its primary function is to use the search tool to find one definitive answer and report it immediately."
    ),
    tools = [tavily_tool], # Assign the Tavily tool to the agent
    llm = llm, # Assign the Groq LLM instance here
    verbose = True, # Show the thought process of the agent
    allow_delegation = False # Since this is a single-agent crew, delegation is not necessary
)

# --- 4. TASK DEFINITION (Simplified for efficiency) ---
research_task = Task(
    description = (
        "Use the search tool to find the current official world record time for the men's 100-meter sprint.\n"
        "Your final answer MUST be only the time and the name of the athlete who holds the record, as a single sentence."
    ),
    expected_output = "A single sentence containing the record time and the athlete's name (e.g., The record is 9.58 seconds, held by Usain Bolt).",
    agent = researcher_agent # Assign the task to the single agent
)

# --- 5. CREW SETUP AND EXECUTION ---
# A Crew is initialized even for a single agent to manage the task execution.
single_agent_crew = Crew(
    agents = [researcher_agent],
    tasks = [research_task],
    verbose = True # Verbose level 2 shows agent step-by-step reasoning
)

# Kick off the research process
print("--- Starting Single-Agent Crew Execution (Powered by Groq) ---")
print("The Researcher will now use the Tavily Search tool to gather information...")

# The result is the final output of the last task
result = single_agent_crew.kickoff()

print("\n\n################################")
print("## CREW EXECUTION FINISHED ##")
print("################################\n")
print(f"Final Research Summary:\n\n{result}")