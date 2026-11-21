"""
Generate a TTS dataset with inclusionAI/Ming-3.0-tts-0.5B.

Run from inside the cloned Ming-3.0-tts repo (so `ming_tts` is importable):
    python make_dataset.py --model_path ./Ming-3.0-tts-0.5b \
        --sentences sentences.txt --out dataset --language English

sentences.txt : one sentence per line.
Output        : dataset/wavs/*.wav (44.1 kHz) + dataset/metadata.csv
                (LJSpeech-style: file|text|speaker)
"""
import argparse
import csv
import os

from ming_tts import MingTTS

# --- Voices to generate with. Edit freely. ---------------------------------
# Built-in voices (task="IP"). Use exact names from the repo's voice catalog.
BUILTIN_VOICES = ["US_James", "US_Stella", "UK_George", "US_Emma"]

# Designed voices (task="VOICE_DESIGN"): name -> text description.
DESIGNED_VOICES = {
    "narrator_male": "A calm, mature male documentary narrator with clear diction and a slow pace.",
    "young_female": "A bright, friendly young female voice, energetic and expressive.",
}

# Cloned voices (task="TTS"): name -> (reference wav, its exact transcript).
# Only clone voices you have permission to use.
CLONED_VOICES = {
    # "my_voice": ("refs/me.wav", "Exact transcript of what I say in me.wav."),
}
# ---------------------------------------------------------------------------


def jobs():
    for v in BUILTIN_VOICES:
        yield v, dict(task="IP", controls={"ip": v}, use_zero_spk_emb=True)
    for name, desc in DESIGNED_VOICES.items():
        yield name, dict(task="VOICE_DESIGN", controls={"style": desc}, use_zero_spk_emb=True)
    for name, (wav, txt) in CLONED_VOICES.items():
        yield name, dict(task="TTS", prompt_wav_path=wav, prompt_text=txt, use_spk_emb=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="inclusionAI/Ming-3.0-tts-0.5B")
    ap.add_argument("--sentences", required=True)
    ap.add_argument("--out", default="dataset")
    ap.add_argument("--language", default="English")
    args = ap.parse_args()

    wav_dir = os.path.join(args.out, "wavs")
    os.makedirs(wav_dir, exist_ok=True)
    meta_path = os.path.join(args.out, "metadata.csv")

    with open(args.sentences, encoding="utf-8") as f:
        sentences = [s.strip() for s in f if s.strip()]

    # Resume support: skip files already generated.
    done = set()
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            done = {row[0] for row in csv.reader(f, delimiter="|")}

    model = MingTTS(args.model_path)  # load once

    with open(meta_path, "a", encoding="utf-8", newline="") as meta:
        writer = csv.writer(meta, delimiter="|")
        for speaker, kwargs in jobs():
            for i, text in enumerate(sentences):
                fname = f"{speaker}_{i:05d}.wav"
                if fname in done:
                    continue
                try:
                    model.synthesize(
                        language_id=args.language,
                        text=text,
                        output_path=os.path.join(wav_dir, fname),
                        **kwargs,
                    )
                    writer.writerow([fname, text, speaker])
                    meta.flush()
                    print("ok", fname)
                except Exception as e:  # keep going on a bad line
                    print("FAILED", fname, e)


if __name__ == "__main__":
    main()
