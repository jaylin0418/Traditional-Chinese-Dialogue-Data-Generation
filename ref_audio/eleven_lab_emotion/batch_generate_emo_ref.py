"""
Batch generate emotional reference WAV files using ElevenLabs API (eleven_v3).

Output naming:
  {emotion}_{speaker}_{variant_num}.wav  →  emotion-specific samples
  {speaker}_{num}.wav                    →  neutral fallback (no emotion)

Output dirs: male/ or female/

26 emotions × 10 speakers × 3 variants = 780 WAV files
+            10 speakers  × 3 variants =  30 neutral fallback WAV files
= 810 WAV files total
"""

import json
import os
import time
import wave
from pathlib import Path
from elevenlabs.client import ElevenLabs

from dotenv import find_dotenv, load_dotenv
load_dotenv(find_dotenv())
client = ElevenLabs(api_key=os.environ["ELEVENLABS_API_KEY"])

BASE_DIR = Path(__file__).parent

# ── All 10 speakers ────────────────────────────────────────────────────────────
SPEAKERS = [
    {"name": "ranbir",  "voice_id": "SGbOfpm28edC83pZ9iGb", "gender": "male"},
    {"name": "roger",   "voice_id": "CwhRBWXzGAHq8TQ4Fs17", "gender": "male"},
    {"name": "charlie", "voice_id": "IKne3meq5aSn9XLyUdCD", "gender": "male"},
    {"name": "george",  "voice_id": "JBFqnCBsd6RMkjVDRZzb", "gender": "male"},
    {"name": "callum",  "voice_id": "N2lVS1w4EtoT3dr4eOWO", "gender": "male"},
    {"name": "river",   "voice_id": "SAz9YHcvj6GT2YYXdXww", "gender": "male"},
    {"name": "harry",   "voice_id": "SOYHLrjzK2X1ezoPC6cr", "gender": "male"},
    {"name": "bella",   "voice_id": "hpp4J3VqNfWAUOO0d1Us", "gender": "female"},
    {"name": "sarah",   "voice_id": "EXAVITQu4vr4xnSDxMaL", "gender": "female"},
    {"name": "laura",   "voice_id": "FGY2WhTYpPnrIDTdsKH5", "gender": "female"},
]

# ── 26 emotions × 3 text variants (Taiwanese Mandarin) ────────────────────────
EMOTIONS: dict[str, list[str]] = {
    "afraid": [
        "我好害怕，感覺有什麼事情要發生了，全身都在發抖，完全不知道該怎麼辦。",
        "那個聲音讓我毛骨悚然，我一步都不敢動，心臟跳得好快。",
        "不知道為什麼，我就是有種很不好的預感，整個人都緊繃起來了。",
    ],
    "amusement": [
        "哈哈哈，這也太好笑了吧！我從來沒看過這麼有趣的事情，真的讓我笑翻了。",
        "你剛才說的那個笑話，我回想起來還是一直噴飯，太好笑了啦。",
        "這個情況也太搞笑了，我完全沒辦法憋住，笑到肚子都痛了。",
    ],
    "angry": [
        "你到底在幹嘛！我已經說了很多次了，你還是繼續這樣，真的讓我很生氣！",
        "講了多少遍了還是搞不懂，你有沒有在聽！我快忍不住了！",
        "這種事情一而再、再而三，你當我都沒有底線嗎！我現在很火大！",
    ],
    "anxiety": [
        "我一直在擔心這件事，腦袋停不下來，昨晚根本睡不著，越想越緊張。",
        "明天的結果不知道會怎樣，我現在心裡七上八下的，完全定不下心來。",
        "感覺事情有點不對勁，我說不上來是哪裡，就是一直有種不安的感覺。",
    ],
    "calm": [
        "沒關係，我們慢慢來，深呼吸，一步一步把問題解決，不用著急。",
        "事情已經發生了，慌也沒有用，冷靜下來才能找到解決辦法。",
        "你放心，這件事我們可以好好處理，不需要緊張，慢慢說。",
    ],
    "compassion": [
        "我了解你現在的感受，這真的很不容易，你已經很努力了，我陪著你。",
        "看到你這樣我心裡也很不是滋味，有什麼需要的你儘管說，我在這裡。",
        "你不用一個人扛這些，我知道你很辛苦，有我在，我們一起面對。",
    ],
    "contentment": [
        "現在這樣就很好了，我很滿足，不需要更多，就這樣剛剛好。",
        "吃飽了、睡好了、身邊有好朋友，這樣的日子我覺得很幸福。",
        "其實不需要什麼大事，平靜地過每一天，我就覺得很知足了。",
    ],
    "cry": [
        "嗚嗚嗚，我真的很難過，眼淚一直流，我忍不住，心裡好痛。",
        "我以為我可以撐住的，但是一想到這件事，就一直哭停不下來。",
        "對不起，我沒辦法不哭，這件事對我來說真的太難了，讓我哭一下好嗎。",
    ],
    "disappointment": [
        "我以為這次會不一樣，結果還是讓我失望了，早知道就不要抱那麼大的期望。",
        "我真的很期待，沒想到最後是這個結果，心裡一時之間難以接受。",
        "算了，反正也不是第一次了，我只是沒想到還是會這麼失望。",
    ],
    "disgusted": [
        "這也太噁心了吧，看到這個我整個反胃，真的完全受不了。",
        "這種行為我完全沒辦法接受，光是想到就覺得好噁，太過分了。",
        "拜託不要讓我再看到那個東西，我已經快吐了，真的太噁心了。",
    ],
    "envy": [
        "為什麼他什麼都順利，什麼都有，而我就什麼都沒有，真的很不公平。",
        "看到他過得那麼好，我心裡說不羨慕是假的，就是會忍不住比較。",
        "同樣的努力，為什麼他得到的就是比我多，我真的有點嫉妒。",
    ],
    "excitement": [
        "太棒了！我好興奮！這真的太厲害了！我等這一天等好久了，終於來了！",
        "我根本睡不著，明天就要去了，光想到就超級期待，心跳好快！",
        "哇！這是真的嗎！我不敢相信，太開心了，我好想大叫出來！",
    ],
    "frustration": [
        "這到底為什麼還沒解決！搞了這麼久還是一樣，我快崩潰了，太令人抓狂了！",
        "試了一次又一次，每次都失敗，我快失去耐心了，到底哪裡出問題！",
        "這個問題明明就不難，但就是過不了，已經卡了好幾個小時，太崩潰了。",
    ],
    "gratitude": [
        "真的非常謝謝你，你幫了我好大的忙，我不知道沒有你的話我要怎麼辦。",
        "你願意在這個時候幫我，我真的很感動，這份情我一定會記住的。",
        "謝謝你一直都在，你對我的好我都放在心裡，我真的很感激你。",
    ],
    "grief": [
        "我失去了一個對我來說很重要的人，心裡有一個洞，怎麼樣都填不滿。",
        "這種失去的感覺太沉重了，我不知道要怎麼繼續走下去，好痛。",
        "每次想到他就忍不住難過，時間過了這麼久，那份悲傷還是在。",
    ],
    "guilt": [
        "都是我的錯，當時我不應該那樣做的，我對不起你，我真的很後悔。",
        "如果我當時做了不同的選擇，也許就不會發生這件事，我一直在責怪自己。",
        "我知道說對不起可能不夠，但我真的很內疚，希望你可以原諒我。",
    ],
    "happy": [
        "今天真的太開心了！事情都很順利，我心情超好，感覺整個人都輕鬆了！",
        "我現在笑得合不攏嘴，這個消息太讓我高興了，真是太棒了！",
        "一整天都過得好順，每件事都往好的方向走，今天真的是美好的一天！",
    ],
    "hope": [
        "我相信只要繼續努力，事情一定會慢慢變好的，我還沒有放棄。",
        "雖然現在很艱難，但我覺得黑暗之後一定有光，我要繼續撐下去。",
        "也許現在看不到結果，但我有預感，只要堅持，一定會看到希望的。",
    ],
    "melancholic": [
        "看著這些舊照片，那些日子再也回不來了，心裡有種說不出來的惆悵。",
        "現在的一切都好陌生，我有時候很懷念以前那段簡單的時光。",
        "不知道為什麼，今天特別感傷，可能是天氣的關係，心裡空空的。",
    ],
    "pride": [
        "我終於做到了，這是我努力很久的成果，我對自己感到很驕傲。",
        "一路走來真的不容易，但我撐過來了，我為自己感到自豪。",
        "這是我用心完成的，不管別人怎麼說，我知道這代表什麼，我很驕傲。",
    ],
    "relief": [
        "終於解決了，我鬆了好大一口氣，之前一直懸在心上，現在終於可以放鬆了。",
        "謝天謝地，結果比我想的好多了，我之前擔心到快喘不過氣，現在好多了。",
        "好加在，沒有發生最壞的情況，我真的放心了，整個人輕鬆了很多。",
    ],
    "sad": [
        "今天心情很低落，什麼都不想做，只想一個人靜靜待著，感覺很空虛。",
        "我說不上來為什麼這麼難過，就是有點悲傷，提不起勁來。",
        "這件事讓我心情很差，我需要一點時間自己靜一靜，心裡好沉。",
    ],
    "shame": [
        "我覺得很丟臉，當時那樣做真的不對，現在根本不知道怎麼面對大家。",
        "想到那件事我就想找個地洞鑽進去，真的很後悔自己當時的行為。",
        "我沒辦法原諒自己，那樣的事怎麼可以做出來，我覺得很羞恥。",
    ],
    "surprised": [
        "哇，你怎麼會在這裡！我完全沒想到，真的嚇到我了，太意外了！",
        "什麼！這件事是真的嗎！我完全沒有預料到，太衝擊了，我需要緩一下。",
        "天啊，我做夢都沒想到會有這個結果，真的太驚訝了，說不出話來。",
    ],
    "sarcastic": [
        "哦，是喔，你真的很厲害呢，我完全沒辦法說什麼，只能佩服你了，真是了不起。",
        "哇，這個方法真是高明，我怎麼就沒想到呢，你真是太聰明了。",
        "對對對，你說的都對，反正什麼事都是你最厲害，我哪裡比得上你呢。",
    ],
    "hysteria": [
        "不行了不行了！我完全不知道該怎麼辦了！腦袋一片空白！到底發生什麼事！",
        "啊啊啊！這怎麼可以這樣！我受不了了！一切都亂掉了！怎麼辦怎麼辦！",
        "全部崩掉了！什麼都失控了！我快瘋掉了！有誰可以幫幫我嗎！",
    ],
}

# ── Neutral fallback texts (3 per speaker, no emotion tag) ───────────────────
NEUTRAL_TEXTS = [
    "請問您需要什麼協助呢？我可以幫您查詢相關資訊或說明辦理流程。",
    "好的，我了解您的需求，讓我幫您確認一下相關內容，請稍等一下。",
    "沒問題，這個問題我來為您處理，如果有任何疑問歡迎繼續詢問。",
]

# ── Per-emotion tag combos: 3 variants, each with different enhancer tags ─────
# Format used in prompt: [strong Taiwanese accent]{tags} text
# Rule: the 3 variants for the same emotion must not be identical tag strings.
EMOTION_TAGS: dict[str, list[str]] = {
    "afraid":         ["[afraid][scared][trembling]",
                       "[afraid][panicked][breathing heavily]",
                       "[afraid][terrified][nervous]"],
    "amusement":      ["[amusement][laughs][cheerful]",
                       "[amusement][giggle][joyful]",
                       "[amusement][starts laughing][happy]"],
    "angry":          ["[angry][shouting][frustrated]",
                       "[angry][tense][assertive]",
                       "[angry][yelling][annoyed]"],
    "anxiety":        ["[anxious][nervous][trembling]",
                       "[anxious][tense][panicked breathing]",
                       "[anxious][scared][stammers]"],
    "calm":           ["[calm][gentle][slowly]",
                       "[calm][softly][content]",
                       "[calm][satisfied][quietly]"],
    "compassion":     ["[empathetic][gentle][concerned]",
                       "[empathetic][softly][calm]",
                       "[empathetic][concerned][quietly]"],
    "contentment":    ["[content][satisfied][calm]",
                       "[content][relieved][gentle]",
                       "[content][blissful][softly]"],
    "cry":            ["[crying][sobs][mournful]",
                       "[crying][wails][sad]",
                       "[crying][tear up][depressed]"],
    "disappointment": ["[disappointed][sighs][sad]",
                       "[disappointed][melancholic][heavy sigh]",
                       "[disappointed][depressed][mournful]"],
    "disgusted":      ["[disgusted][annoyed][groans]",
                       "[disgusted][frustrated][tense]",
                       "[disgusted][angry][grunts]"],
    "envy":           ["[envy][frustrated][tense]",
                       "[envy][disappointed][annoyed]",
                       "[envy][melancholic][sad]"],
    "excitement":     ["[excited][energetic][joyful]",
                       "[excited][cheerful][fast]",
                       "[excited][happy][shouting]"],
    "frustration":    ["[frustrated][tense][groans]",
                       "[frustrated][annoyed][fast]",
                       "[frustrated][angry][stammers]"],
    "gratitude":      ["[grateful][happy][gentle]",
                       "[grateful][relieved][calm]",
                       "[grateful][content][softly]"],
    "grief":          ["[mournful][crying][slowly]",
                       "[mournful][sobs][depressed]",
                       "[mournful][sad][heavy sigh]"],
    "guilt":          ["[guilt][sad][quietly]",
                       "[guilt][depressed][stammers]",
                       "[guilt][disappointed][softly]"],
    "happy":          ["[happy][joyful][cheerful]",
                       "[happy][excited][energetic]",
                       "[happy][blissful][optimistic]"],
    "hope":           ["[optimistic][calm][gentle]",
                       "[optimistic][content][slowly]",
                       "[optimistic][relieved][softly]"],
    "melancholic":    ["[melancholic][sad][slowly]",
                       "[melancholic][mournful][sighs]",
                       "[melancholic][depressed][quietly]"],
    "pride":          ["[pride][confident][assertive]",
                       "[pride][satisfied][calm]",
                       "[pride][content][slowly]"],
    "relief":         ["[relieved][sighs][calm]",
                       "[relieved][content][exhales]",
                       "[relieved][satisfied][heavy sigh]"],
    "sad":            ["[sad][mournful][slowly]",
                       "[sad][depressed][sighs]",
                       "[sad][melancholic][quietly]"],
    "shame":          ["[shame][depressed][quietly]",
                       "[shame][sad][stammers]",
                       "[shame][disappointed][softly]"],
    "surprised":      ["[surprised][shocked][gasps]",
                       "[surprised][excited][fast]",
                       "[surprised][energetic][shouting]"],
    "sarcastic":      ["[sarcastic][calm][slowly]",
                       "[sarcastic][assertive][confident]",
                       "[sarcastic][annoyed][quietly]"],
    "hysteria":       ["[panicked][shouting][fast]",
                       "[terrified][yelling][trembling]",
                       "[panicked][breathing heavily][stammers]"],
}

SAMPLE_RATE = 22050
NUM_CHANNELS = 1
SAMPLE_WIDTH = 2  # 16-bit


def pcm_to_wav(pcm_data: bytes, out_path: Path):
    with wave.open(str(out_path), "wb") as wf:
        wf.setnchannels(NUM_CHANNELS)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm_data)


def generate_one(tags: str | None, text: str, speaker: dict, out_path: Path, retries: int = 3) -> bool:
    prompt = f"[strong Taiwanese accent]{tags} {text}" if tags else f"[strong Taiwanese accent] {text}"
    for attempt in range(1, retries + 1):
        try:
            audio_iter = client.text_to_speech.convert(
                voice_id=speaker["voice_id"],
                model_id="eleven_v3",
                text=prompt,
                language_code="zh",
                output_format="pcm_22050",
            )
            pcm_data = b"".join(audio_iter)
            pcm_to_wav(pcm_data, out_path)
            return True
        except Exception as e:
            print(f"  [attempt {attempt}/{retries}] ERROR: {e}")
            if attempt < retries:
                time.sleep(5 * attempt)
    return False


def main():
    for gender in ("male", "female"):
        (BASE_DIR / gender).mkdir(exist_ok=True)

    neutral_count = len(SPEAKERS) * len(NEUTRAL_TEXTS)
    emotion_count = len(EMOTIONS) * len(SPEAKERS) * 3
    total = emotion_count + neutral_count
    done = 0
    failed = []

    # ── Emotion samples ────────────────────────────────────────────────────────
    for speaker in SPEAKERS:
        out_dir = BASE_DIR / speaker["gender"]
        for emotion, texts in EMOTIONS.items():
            for variant_idx, text in enumerate(texts, start=1):
                done += 1
                out_path = out_dir / f"{emotion}_{speaker['name']}_{variant_idx}.wav"

                if out_path.exists():
                    print(f"[{done:>3}/{total}] SKIP : {out_path.name}")
                    continue

                tags = EMOTION_TAGS[emotion][variant_idx - 1]
                print(f"[{done:>3}/{total}] GEN  : {out_path.name} ...", end=" ", flush=True)
                ok = generate_one(tags, text, speaker, out_path)
                print("OK" if ok else "FAILED")
                if not ok:
                    failed.append(str(out_path))
                time.sleep(0.4)

    # ── Neutral fallback samples ───────────────────────────────────────────────
    for speaker in SPEAKERS:
        out_dir = BASE_DIR / speaker["gender"]
        for variant_idx, text in enumerate(NEUTRAL_TEXTS, start=1):
            done += 1
            out_path = out_dir / f"{speaker['name']}_{variant_idx}.wav"

            if out_path.exists():
                print(f"[{done:>3}/{total}] SKIP : {out_path.name}")
                continue

            print(f"[{done:>3}/{total}] GEN  : {out_path.name} ...", end=" ", flush=True)
            ok = generate_one(None, text, speaker, out_path)
            print("OK" if ok else "FAILED")
            if not ok:
                failed.append(str(out_path))
            time.sleep(0.4)

    print(f"\nDone. {total - len(failed)}/{total} files generated successfully.")
    if failed:
        print(f"\nFailed ({len(failed)}):")
        for f in failed:
            print(f"  {f}")

    # ── Write transcriptions.json ──────────────────────────────────────────────
    # Maps relative wav path (e.g. "male/happy_george_1.wav") → spoken text.
    # The pipeline uses this to populate speaker_prompt_text_transcription.
    transcriptions: dict[str, str] = {}
    for speaker in SPEAKERS:
        out_dir = BASE_DIR / speaker["gender"]
        for emotion, texts in EMOTIONS.items():
            for variant_idx, text in enumerate(texts, start=1):
                rel = f"{speaker['gender']}/{emotion}_{speaker['name']}_{variant_idx}.wav"
                tags = EMOTION_TAGS[emotion][variant_idx - 1]
                transcriptions[rel] = f"[strong Taiwanese accent]{tags} {text}"
        for variant_idx, text in enumerate(NEUTRAL_TEXTS, start=1):
            rel = f"{speaker['gender']}/{speaker['name']}_{variant_idx}.wav"
            transcriptions[rel] = f"[strong Taiwanese accent] {text}"

    trans_path = BASE_DIR / "transcriptions.json"
    with trans_path.open("w", encoding="utf-8") as f:
        json.dump(transcriptions, f, ensure_ascii=False, indent=2)
    print(f"Saved transcriptions: {trans_path}")


if __name__ == "__main__":
    main()
