import random
from dotenv import load_dotenv
import guidance
from guidance import models, gen, system, user, assistant

load_dotenv()

# set the default language model that execute guidance programs
lm = models.OpenAI("gpt-3.5-turbo")

quizflavor = [
    "Quiz of the day!",
    "Test your knowledge!",
    "Here is a quiz!",
    "You think you know Unix?",
]

def program(lm):
    with system():
        lm += "You are a Linux systems expert creating educational content about command line tools."
    
    with user():
        lm += "What are the top ten most common commands used in the Linux operating system? Provide a one-liner description for each command."
    
    with assistant():
        lm += gen('example', max_tokens=20, temperature=0.8)
    
    with user():
        lm += "Here are the common commands:"
    
    with assistant():
        lm += gen('commands', max_tokens=500)
    
    with user():
        flavor = random.choice(quizflavor)
        points = random.randint(1, 5)
        lm += f"{flavor} Explain the following commands for {points} points:"
    
    with assistant():
        lm += gen('confusing_commands', max_tokens=300)
    
    return lm

result = program(lm)

print("Generated example:")
print(result.get("example", "No example generated"))
print("===")
print("Generated commands:")
print(result.get("commands", "No commands generated"))
print("===")
print("Confusing commands:")
print(result.get("confusing_commands", "No confusing commands generated"))
print("===")
print("Full result:")
print(result)
