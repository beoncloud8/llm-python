from dotenv import load_dotenv
import os

load_dotenv()

from openai import OpenAI
import webbrowser

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# list all models
try:
    models = client.models.list()
    print(models.data[0].id)
    print([model.id for model in models.data])
except Exception as e:
    print(f"Error listing models: {e}")

# create our completion (modern syntax)
try:
    completion = client.completions.create(model="gpt-3.5-turbo-instruct", prompt="Bill Gates is a")
    print(completion.choices[0].text)
except Exception as e:
    print(f"Error creating completion: {e}")

# image generation (modern syntax)
try:
    image_gen = client.images.generate(
        prompt="Zwei Hunde spielen unter einem Baum, cartoon",
        n=2,
        size="512x512"
    )
    for img in image_gen.data:
        webbrowser.open_new_tab(img.url)
except Exception as e:
    print(f"Error generating images: {e}")

# Audio transcription (modern syntax)
try:
    with open("audio/donda.mp3", "rb") as audio:
        transcript = client.audio.transcriptions.create(
            model="whisper-1", 
            file=audio
        )
        print(transcript.text)
except Exception as e:
    print(f"Error transcribing audio: {e}")
