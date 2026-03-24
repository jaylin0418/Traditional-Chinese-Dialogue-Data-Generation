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
    {"name": "river",   "voice_id": "SAz9YHcvj6GT2YYXdXww", "gender": "female"},
    {"name": "harry",   "voice_id": "SOYHLrjzK2X1ezoPC6cr", "gender": "male"},
    {"name": "bella",   "voice_id": "hpp4J3VqNfWAUOO0d1Us", "gender": "female"},
    {"name": "sarah",   "voice_id": "EXAVITQu4vr4xnSDxMaL", "gender": "female"},
    {"name": "laura",   "voice_id": "FGY2WhTYpPnrIDTdsKH5", "gender": "female"},
]

# ── 26 emotions × 3 text variants (Taiwanese Mandarin) ────────────────────────
EMOTIONS = {
    "afraid": [
        "我好害怕，感覺有不好的事要發生，全身發抖，腦袋一片空白，不知道該怎麼辦。",
        "那個聲音讓我毛骨悚然，我不敢動，心臟跳很快。我不敢往那邊看，只想快點離開，腳卻像被釘住。",
        "不知道為什麼，我有種不好的預感，整個人都緊繃。每個細小聲音都讓我不安，不知道會發生什麼。",
    ],
    "amusement": [
        "哈哈哈，這也太好笑了吧！我沒看過這麼有趣的事，真的笑翻了，完全停不下來！",
        "你剛才那個笑話，我現在想起來還是想笑。每次想到那表情，真的會忍不住噴飯。",
        "這情況也太搞笑了，我真的完全憋不住，笑到肚子都痛了，實在太有趣了。",
    ],
    "angry": [
        "你到底在幹嘛！我都說很多次了，你還這樣，真的讓我很生氣！你有在聽嗎？",
        "講了多少遍你還是不懂，你到底有沒有在聽！這件事我講很清楚了，能不能解釋一下？",
        "這種事一再發生，你當我沒有底線嗎！我忍很久了，不要再逼我，我真的很火大！",
    ],
    "anxiety": [
        "我一直在擔心這件事，腦袋停不下來，昨晚也睡不好，越想越緊張，完全冷靜不了。",
        "明天的結果不知道會怎樣，我現在心裡七上八下，做什麼都無法放鬆，真的很焦慮。",
        "感覺事情有點不對，我說不上哪裡有問題，就是一直不安，心裡怎麼樣都靜不下來。",
    ],
    "calm": [
        "沒關係，我們慢慢來，先深呼吸，一步一步把問題理清，不用急，我們一起想辦法。",
        "事情都發生了，慌也沒用，先冷靜下來。我們把問題一件件整理，這樣比較好處理。",
        "你放心，這件事可以慢慢處理，不需要緊張。只要保持冷靜，就一定找得到方法。",
    ],
    "compassion": [
        "我知道你現在很不好受，這真的不容易，你已經很努力了，不要再太為難自己了。",
        "看到你這樣，我心裡也很難受。有什麼需要就跟我說，我會陪你一起度過這段時間。",
        "你不用一個人扛這些，我知道你很辛苦。有我在，我們一起面對，你真的不孤單。",
    ],
    "contentment": [
        "現在這樣就很好了，我很滿足，不需要更多。能安穩過日子，對我來說就很幸福了。",
        "吃飽睡好，身邊也有朋友，這樣的日子我就覺得很幸福，不需要再多求什麼了。",
        "其實不用發生什麼大事，能平靜過每一天，我就很知足，簡單生活就是幸福。",
    ],
    "cry": [
        "嗚嗚，我真的很難過，眼淚一直流，心裡好痛，我不知道怎麼辦，讓我先哭一下。",
        "我以為自己可以撐住，但一想到這件事就一直哭，真的停不下來，我真的很難過。",
        "對不起，我真的沒辦法不哭，這件事對我太難了，讓我哭一下，等一下再跟你說。",
    ],
    "disappointment": [
        "我以為這次會不一樣，結果還是讓我失望了。早知道就不要抱那麼大的期待。",
        "我真的很期待，沒想到最後是這結果，心裡很難接受，也不知道該怎麼面對才好。",
        "算了，反正也不是第一次了，只是沒想到還是會這麼失望，真的很難不難過。",
    ],
    "disgusted": [
        "這也太噁心了吧，看到我整個反胃，真的完全受不了，根本不想再多看一眼。",
        "這種行為我真的無法接受，光想到就覺得很噁心，你怎麼可以做出這種事。",
        "拜託不要再讓我看到那東西了，我真的快吐了，太噁心了，讓人很不舒服。",
    ],
    "envy": [
        "為什麼他什麼都順利，什麼都有，而我卻什麼都沒有，真的讓人很不是滋味。",
        "看到他過得那麼好，我說不羨慕是假的，就是會忍不住比較，心裡真的很複雜。",
        "同樣努力，為什麼他得到的就是比我多，我真的有點嫉妒，心裡很不平衡。",
    ],
    "excitement": [
        "太棒了！我好興奮！我等這一天等很久了，現在終於來了，真的開心到不行！",
        "我根本睡不著，明天就要去了，光想到就超期待，心跳一直很快，真的太興奮了！",
        "哇！這是真的嗎！我不敢相信，真的太開心了，好想直接大叫出來，太爽了！",
    ],
    "frustration": [
        "這到底為什麼還沒解決！搞了這麼久還是一樣，我真的快崩潰了，到底哪裡有問題！",
        "試了一次又一次，每次都失敗，我快失去耐心了。再這樣下去，我真的想放棄了。",
        "這問題明明不難，但就是過不了，已經卡好幾小時了，每次快好了又出新狀況。",
    ],
    "gratitude": [
        "真的非常謝謝你，你幫了我很大的忙。沒有你的話，我真的不知道該怎麼辦才好。",
        "你願意在這時候幫我，我真的很感動。能有你這樣的朋友，我真的覺得很幸運。",
        "謝謝你一直都在，你對我的好我都記得。每次遇到困難，你總是最先出現的人。",
    ],
    "grief": [
        "我失去了一個很重要的人，心裡像破了一個洞，那種痛真的讓我很難走下去。",
        "這種失去的感覺太沉重了，我不知道要怎麼繼續下去，心裡真的非常痛苦。",
        "每次想到他我都還是會難過，那份悲傷一直都在，我真的很想再見他一面。",
    ],
    "guilt": [
        "都是我的錯，當時我不該那樣做的。我真的很後悔，也不知道該怎麼彌補才好。",
        "如果我當時做了不同選擇，也許就不會變成這樣。我一直責怪自己，真的很痛苦。",
        "我知道一句對不起可能不夠，但我真的很內疚。希望你有一天能原諒我。",
    ],
    "happy": [
        "今天真的太開心了！事情都很順利，心情超好，整個人都輕鬆起來，真的太棒了！",
        "我現在笑得合不攏嘴，這消息真的太讓人高興了，今天真的是很美好的一天！",
        "一整天都過得很順，每件事都往好方向走，今天真的是讓人很開心的一天！",
    ],
    "hope": [
        "我相信只要繼續努力，事情一定會慢慢變好，我還沒有放棄，還想再撐一下。",
        "雖然現在很艱難，但我相信黑暗後面一定有光，所以我還是想繼續撐下去。",
        "也許現在還看不到結果，但我相信只要不放棄，總有一天一定會看到希望。",
    ],
    "melancholic": [
        "看著這些舊照片，那些日子再也回不來了，心裡有種說不出的惆悵，很懷念。",
        "現在的一切都好陌生，我常常想起以前那段簡單時光，真的很想回到那時候。",
        "不知道為什麼，今天特別感傷，心裡空空的，什麼都提不起勁，只想自己待著。",
    ],
    "pride": [
        "我終於做到了，這是我努力很久的成果，我真的為自己感到很驕傲，也很欣慰。",
        "一路走來真的不容易，但我還是撐過來了。現在回頭看，我真的替自己感到自豪。",
        "這是我用心完成的成果，不管別人怎麼說，我知道它的價值，我真的很驕傲。",
    ],
    "relief": [
        "終於解決了，我鬆了好大一口氣。之前一直懸著的心，現在總算可以放下來了。",
        "謝天謝地，結果比我想的好多了。我之前擔心到不行，現在終於能安心一點了。",
        "好加在，沒有發生最壞的情況，我真的放心了，整個人也輕鬆了不少。",
    ],
    "sad": [
        "今天心情很低落，什麼都不想做，只想一個人靜靜待著，心裡真的很難過。",
        "我說不上來為什麼這麼難過，就是提不起勁，什麼都不想碰，只想安靜一下。",
        "這件事讓我心情很差，我想先自己靜一靜。現在心裡很沉，真的有點撐不住。",
    ],
    "shame": [
        "我覺得很丟臉，當時那樣做真的不對，現在根本不知道該怎麼面對大家才好。",
        "想到那件事我就想找地洞鑽進去，真的很後悔自己當時的行為，太丟臉了。",
        "我沒辦法原諒自己，那樣的事怎麼做得出來，我真的覺得非常羞恥。",
    ],
    "surprised": [
        "哇，你怎麼會在這裡！我完全沒想到，真的嚇到我了，太意外了，也太巧了吧！",
        "什麼！這件事是真的嗎！我完全沒預料到，太衝擊了，我真的需要緩一下。",
        "天啊，我做夢都沒想到會有這結果，真的太驚訝了，我現在還回不過神來。",
    ],
    "sarcastic": [
        "哦，是喔，你真的很厲害呢，我完全沒話說了，只能佩服你，真是太了不起了。",
        "哇，這方法真高明，我怎麼就沒想到呢，你真是太聰明了，實在讓人佩服。",
        "對對對，你說的都對，反正什麼事都是你最厲害，我哪裡比得上你呢。",
    ],
    "hysteria": [
        "不行了不行了！我完全不知道該怎麼辦！腦袋一片空白！到底發生什麼事了！",
        "啊啊啊！這怎麼可以這樣！我受不了了！一切都亂掉了！現在到底該怎麼辦！",
        "全部崩掉了！什麼都失控了！我快瘋掉了！誰可以幫幫我，我真的不行了！",
    ],
}

# ── Neutral fallback texts (3 per speaker, no emotion tag) ───────────────────
NEUTRAL_TEXTS = [
    "請問您需要什麼協助呢？我可以幫您查詢資訊或說明流程，有問題都可以直接告訴我。",
    "好的，我了解您的需求，讓我先幫您確認相關內容。請稍等一下，我整理後回覆您。",
    "沒問題，這個問題我來幫您處理。如果還有其他疑問，也歡迎您繼續提出來。",
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

MIN_DURATION = 5.01
MAX_DURATION = 14.99


def get_wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as wf:
        return wf.getnframes() / wf.getframerate()


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


def generate_with_duration_check(
    tags: str | None,
    text: str,
    extra_texts: list[str],
    speaker: dict,
    out_path: Path,
) -> tuple[bool, float]:
    """Generate audio and retry with progressively more text if duration < MIN_DURATION.
    For duration > MAX_DURATION (likely API glitch), retry up to 3 times with same text.
    Returns (success, duration).
    """
    # Build text candidates: original → original+1 extra → original+2 extras
    text_candidates = [text]
    combined = text
    for extra in extra_texts:
        combined = combined + extra
        text_candidates.append(combined)

    for candidate_text in text_candidates:
        # For too-long glitches, retry the same candidate up to 3 times
        for glitch_retry in range(3):
            ok = generate_one(tags, candidate_text, speaker, out_path)
            if not ok:
                return False, 0.0
            dur = get_wav_duration(out_path)
            if MIN_DURATION <= dur <= MAX_DURATION:
                return True, dur
            if dur > MAX_DURATION:
                # API glitch producing absurdly long audio — retry same text
                print(f"\n    [TOO LONG: {dur:.1f}s, retrying same text #{glitch_retry + 1}]", end=" ", flush=True)
                out_path.unlink()
                time.sleep(2)
                continue
            # Too short — break inner loop to try longer candidate
            out_path.unlink()
            break
        else:
            # Exhausted glitch retries, still too long; move to next candidate (shorter won't help, but try)
            continue
        if out_path.exists():
            return True, get_wav_duration(out_path)
        # Was too short — continue to next (longer) candidate

    # Last resort: keep whatever was generated even if out of range
    if out_path.exists():
        return True, get_wav_duration(out_path)
    return False, 0.0


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
                    dur = get_wav_duration(out_path)
                    if MIN_DURATION <= dur <= MAX_DURATION:
                        print(f"[{done:>3}/{total}] SKIP : {out_path.name} ({dur:.2f}s)")
                        continue
                    print(f"[{done:>3}/{total}] REGEN: {out_path.name} (was {dur:.2f}s, out of range)")
                    out_path.unlink()

                tags = EMOTION_TAGS[emotion][variant_idx - 1]
                # Extra texts: other variants for the same emotion (as fallback for too-short)
                extra_texts = [t for i, t in enumerate(texts) if i != variant_idx - 1]
                print(f"[{done:>3}/{total}] GEN  : {out_path.name} ...", end=" ", flush=True)
                ok, dur = generate_with_duration_check(tags, text, extra_texts, speaker, out_path)
                if ok:
                    in_range = MIN_DURATION <= dur <= MAX_DURATION
                    print(f"OK ({dur:.2f}s)" + ("" if in_range else " [OUT OF RANGE]"))
                    if not in_range:
                        failed.append(str(out_path))
                else:
                    print("FAILED")
                    failed.append(str(out_path))
                time.sleep(0.4)

    # ── Neutral fallback samples ───────────────────────────────────────────────
    for speaker in SPEAKERS:
        out_dir = BASE_DIR / speaker["gender"]
        for variant_idx, text in enumerate(NEUTRAL_TEXTS, start=1):
            done += 1
            out_path = out_dir / f"{speaker['name']}_{variant_idx}.wav"

            if out_path.exists():
                dur = get_wav_duration(out_path)
                if MIN_DURATION <= dur <= MAX_DURATION:
                    print(f"[{done:>3}/{total}] SKIP : {out_path.name} ({dur:.2f}s)")
                    continue
                print(f"[{done:>3}/{total}] REGEN: {out_path.name} (was {dur:.2f}s, out of range)")
                out_path.unlink()

            # Extra texts: other neutral variants as fallback
            extra_texts = [t for i, t in enumerate(NEUTRAL_TEXTS) if i != variant_idx - 1]
            print(f"[{done:>3}/{total}] GEN  : {out_path.name} ...", end=" ", flush=True)
            ok, dur = generate_with_duration_check(None, text, extra_texts, speaker, out_path)
            if ok:
                in_range = MIN_DURATION <= dur <= MAX_DURATION
                print(f"OK ({dur:.2f}s)" + ("" if in_range else " [OUT OF RANGE]"))
                if not in_range:
                    failed.append(str(out_path))
            else:
                print("FAILED")
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
