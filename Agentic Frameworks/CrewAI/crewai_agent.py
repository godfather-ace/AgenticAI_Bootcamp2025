#pip install crewai litellm

from crewai import Agent, Task, Crew, LLM, Process
from dotenv import load_dotenv
import os

load_dotenv()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

llm = LLM(model = "groq/llama-3.1-8b-instant", max_completion_tokens = 256)

planner = Agent(
    role = "Content Planner",
    goal = "Agent is able to plan an engaging and factually correct content on {topic}",
    backstory = """You have to plan a linkedin post on the: {topic}
                    Your job is to collect information that is simple for audience
                    and your work is going to be the basis for writer agent for writing the post.""",
    llm = llm, 
    verbose = True
)

writer = Agent(
    role = "Content Writer",
    goal = "Agent is able to write a post about the topic: {topic}",
    backstory = """You have to write a linkedin post on the: {topic}
                    Your writing is based on the work of content planner agent, 
                    which provides you the outline""",
    llm = llm, 
    verbose = True
)

plan = Task(
    description = (
        "1. Get the information on the topic: {topic}" 
        "2. Plan a proper content around the topic"
    ), 
    expected_output = "A content plan based on proper information",
    agent = planner
)

write = Task(
    description = (
        "1. Use the content plan to craft a linkedin post on the topic: {topic}" 
    ), 
    expected_output = "A good written post on the topic",
    agent = writer
)

crew = Crew(
    agents = [planner, writer],
    tasks = [plan, write],
    verbose = True,
    tracing = True
)

result = crew.kickoff(inputs = {"topic": "Large Language Models"})
print(result)

