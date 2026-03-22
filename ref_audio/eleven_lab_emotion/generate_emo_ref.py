import os
from dotenv import find_dotenv, load_dotenv
from elevenlabs.client import ElevenLabs

load_dotenv(find_dotenv())
client = ElevenLabs(api_key=os.environ["ELEVENLABS_API_KEY"])




voices = client.voices.search()


for v in voices.voices:
    print(f"name={v.name} | voice_id={v.voice_id}")


audio = client.text_to_speech.convert(
    voice_id="JBFqnCBsd6RMkjVDRZzb",
    model_id="eleven_v3",
    text="[Taiwanese , happy] 我是台灣人。你現在最好給我說清楚，為什麼事情會搞成這樣。我不是沒給過你機會，是你一次又一次把我的信任踩在地上，真的不要逼我翻臉。",
    language_code="zh",
    output_format="mp3_22050_32",
)

with open("wo_ai_ni.mp3", "wb") as f:
    for chunk in audio:
        f.write(chunk)

print("saved: wo_ai_ni.mp3")