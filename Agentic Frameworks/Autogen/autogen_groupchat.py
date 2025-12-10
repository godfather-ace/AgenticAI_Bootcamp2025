import autogen
import os
from dotenv import load_dotenv

try: 
    load_dotenv()
    llm_config = {
        "config_list": [
            {
                "model": "llama-3.1-8b-instant",
                "api_key": os.getenv("GROQ_API_KEY"),
                "api_type": "groq"
            }
        ]
    }
except Exception as e: 
    print("Error Loading config: {e}")
    exit()

user_proxy = autogen.UserProxyAgent(
    name = "Admin", 
    system_message = "A human admin. Send comments and terminate the chat by saying 'TERMINATE when the task is fully complete.",
    code_execution_config = {
        "work_dir": "group_chat_work",
        "use_docker": False
    }, 
    is_termination_msg = lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
    human_input_mode = "NEVER"
)

coder = autogen.AssistantAgent(
    name = "Coder", 
    llm_config = llm_config, 
    system_message = "You are a professional Python programmer. Your job is to write high-quality Python codes. You must wait for the Critic's approval before executing or confirming the final code.", 
)

critic = autogen.AssistantAgent(
    name = "Critic",
    llm_config = llm_config, 
    system_message = "You are a code tester. Your job is to check the Coder's plan and code for correctness, reliability. You provide constructive criticism and must explicitely give the final approval ('LOOKS GOOD!') only when the code is perfect and ready."
)

group_chat = autogen.GroupChat(
    agents = [user_proxy, coder, critic],
    messages = [],
    max_round = 7,
    allow_repeat_speaker = False
)

manager = autogen.GroupChatManager(
    groupchat = group_chat, 
    llm_config = llm_config, 
    system_message = "You are the Group Chat Manager. You moderate the chat between Coder and Critic, making sure that the Coder provides code and the Critic reviews it before the Admin approves the final solution."
)

user_proxy.initiate_chat(
    manager, 
    message = "Write a Python script to calculate the first 10 fibonacci numbers. The script must then save the complete code to a file named 'demo.txt'. The Coder must get the approval from Critic before executing or proposing the final solution."
)

print(f"Check the '{os.path.join(os.getcwd(), 'group_chat_work')}' directory for the outputfile 'demo.txt'")