"""
Check which emotional reference audio files have duration outside [6, 14] seconds.
Scans both female/ and male/ subdirectories.
"""

import wave
from pathlib import Path

BASE_DIR = Path(__file__).parent
MIN_SEC = 6.0
MAX_SEC = 12.0

def get_wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as f:
        frames = f.getnframes()
        rate = f.getframerate()
        return frames / rate

TXT_OUT = BASE_DIR / "out_of_range_audio.txt"


def main():
    out_of_range = []
    all_files = []

    for subdir in ["female", "male"]:
        for wav_path in sorted((BASE_DIR / subdir).glob("*.wav")):
            all_files.append(wav_path)
            try:
                duration = get_wav_duration(wav_path)
            except Exception as e:
                print(f"[ERROR] {wav_path.relative_to(BASE_DIR)}: {e}")
                continue

            if not (MIN_SEC <= duration <= MAX_SEC):
                out_of_range.append((wav_path.relative_to(BASE_DIR), duration))

    lines = []
    lines.append(f"Scanned {len(all_files)} WAV files.")
    lines.append("")

    if not out_of_range:
        lines.append(f"All files are within [{MIN_SEC}, {MAX_SEC}] seconds.")
    else:
        lines.append(f"Files outside [{MIN_SEC}, {MAX_SEC}] seconds ({len(out_of_range)} total):")
        lines.append(f"{'File':<55} {'Duration (s)':>12}  Flag")
        lines.append("-" * 75)
        for rel_path, dur in out_of_range:
            flag = "TOO SHORT" if dur < MIN_SEC else "TOO LONG"
            lines.append(f"{str(rel_path):<55} {dur:>10.2f}s  [{flag}]")

    output = "\n".join(lines)
    print(output)

    with TXT_OUT.open("w", encoding="utf-8") as f:
        f.write(output + "\n")
    print(f"\nResults written to: {TXT_OUT}")

if __name__ == "__main__":
    main()
