from dotenv import load_dotenv
import guidance
from guidance import models, gen, system, user, assistant

load_dotenv()

# Updated to use the new API syntax
chat = models.OpenAI("gpt-3.5-turbo")

def program(lm, os):
    with system():
        lm += f"You are a CS Professor teaching {os} systems administration to your students."
    
    with user():
        lm += f"What are some of the most common commands used in the {os} operating system? Provide a one-liner description. List the commands and their descriptions one per line. Number them starting from 1."
    
    with assistant():
        lm += gen('commands', max_tokens=100)
    
    with user():
        lm += "Which among these commands are beginners most likely to get wrong? Explain why the command might be confusing. Show example code to illustrate your point."
    
    with assistant():
        lm += gen('confusing_commands', max_tokens=100)
    
    return lm

result = program(chat, os="Linux")

print(result["commands"])
print("===")
print(result)
