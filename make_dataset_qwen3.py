"""
Build a multi-speaker TTS dataset with Qwen3-TTS (local ModelScope folders).

Speakers:
  1. Preset speakers  - CustomVoice model (Ryan, Aiden, Vivian, ...), each line
                         read with a random speaking style.
  2. Designed voices  - N random voice descriptions (gender, age, pitch, ...).
       * With --base (recommended): each voice is designed ONCE into a reference
         clip, then the Base model clones it for every sentence, so a speaker
         sounds the same across the whole dataset.
       * Without --base: every line is voice-designed from the description
         directly. Faster to set up, but the "same" speaker can drift between lines.

Output:
  OUT/wavs/<speaker>_<idx>.wav
  OUT/metadata.csv     file|text|speaker|style       (LJSpeech-style, '|' separated)
  OUT/voices.json      description of every designed voice
  OUT/voice_refs/      reference clips of designed voices (with --base)

Example:
  export HF_HUB_OFFLINE=1
  python make_dataset_qwen3.py \
      --sentences sentences.txt --out dataset --language English \
      --custom ./Qwen3-TTS-12Hz-1.7B-CustomVoice \
      --design ./Qwen3-TTS-12Hz-1.7B-VoiceDesign \
      --base   ./Qwen3-TTS-12Hz-1.7B-Base \
      --num_designed 40 --batch_size 8

Rerunning the same command resumes: finished files are skipped.
"""
import argparse
import csv
import json
import os
import random

import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel

# ---------------------------------------------------------------- settings ---
# Preset speakers. Leave empty to use every speaker the CustomVoice model reports.
PRESET_SPEAKERS = []  # e.g. ["Ryan", "Aiden"] for English-only

# Styles mixed into preset-speaker lines ("" = neutral, no instruction).
PRESET_STYLES = [
    "",
    "Speak in a cheerful, upbeat tone.",
    "Speak calmly and slowly.",
    "Speak with a sad, quiet voice.",
    "Speak with excitement and energy.",
    "Speak in a serious, formal newsreader tone.",
    "Speak in a warm, friendly conversational tone.",
    "Speak a little angrily, with frustration.",
]

# Attribute pools for random designed voices.
GENDERS = ["male", "female"]
AGES = ["teenage", "young adult", "adult in their thirties", "middle-aged", "elderly"]
PITCHES = ["low-pitched", "medium-pitched", "high-pitched"]
PACES = ["slow", "moderate", "fast"]
TIMBRES = ["warm", "bright", "husky", "smooth", "deep and resonant", "slightly nasal",
           "soft and breathy", "crisp and clear", "gravelly"]
ACCENTS = ["a neutral American accent", "a British accent", "an Australian accent",
           "an Indian English accent", "a Scottish accent", "a neutral accent"]
MOODS = ["calm", "friendly", "confident", "energetic", "gentle", "serious", "cheerful"]

# Fixed sentence used to build each designed voice's reference clip.
REF_TEXT = ("Hello, it is nice to meet you. Today I will read a few sentences "
            "for you in my natural speaking voice.")
# ----------------------------------------------------------------------------


def random_voice(rng):
    return (f"A {rng.choice(MOODS)} {rng.choice(AGES)} {rng.choice(GENDERS)} voice, "
            f"{rng.choice(PITCHES)} and {rng.choice(TIMBRES)}, speaking at a "
            f"{rng.choice(PACES)} pace with {rng.choice(ACCENTS)}.")


def load(path):
    return Qwen3TTSModel.from_pretrained(path, device_map="cuda:0", dtype=torch.bfloat16)


def chunks(items, n):
    for i in range(0, len(items), n):
        yield items[i:i + n]


class Writer:
    """Writes wavs + metadata rows, remembers what is already done (resume)."""

    def __init__(self, out):
        self.wav_dir = os.path.join(out, "wavs")
        os.makedirs(self.wav_dir, exist_ok=True)
        self.meta_path = os.path.join(out, "metadata.csv")
        self.done = set()
        if os.path.exists(self.meta_path):
            with open(self.meta_path, encoding="utf-8") as f:
                self.done = {row[0] for row in csv.reader(f, delimiter="|") if row}
        self.f = open(self.meta_path, "a", encoding="utf-8", newline="")
        self.w = csv.writer(self.f, delimiter="|")

    def todo(self, speaker, sentences):
        return [(i, t) for i, t in enumerate(sentences)
                if f"{speaker}_{i:05d}.wav" not in self.done]

    def save(self, speaker, idx, text, style, wav, sr):
        name = f"{speaker}_{idx:05d}.wav"
        sf.write(os.path.join(self.wav_dir, name), wav, sr)
        self.w.writerow([name, text, speaker, style])
        self.f.flush()
        self.done.add(name)
        print("ok", name)


def run_batches(writer, speaker, jobs, batch_size, gen_fn):
    """jobs: list of (idx, text, style). gen_fn(texts, styles) -> (wavs, sr)."""
    for batch in chunks(jobs, batch_size):
        idxs, texts, styles = zip(*batch)
        try:
            wavs, sr = gen_fn(list(texts), list(styles))
            for i, t, s, w in zip(idxs, texts, styles, wavs):
                writer.save(speaker, i, t, s, w, sr)
        except Exception as e:  # one bad batch shouldn't kill a long run
            print(f"FAILED {speaker} batch starting {idxs[0]}: {e}")
            torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sentences", required=True, help="text file, one sentence per line")
    ap.add_argument("--out", default="dataset")
    ap.add_argument("--language", default="English")
    ap.add_argument("--custom", help="path to Qwen3-TTS-12Hz-1.7B-CustomVoice")
    ap.add_argument("--design", help="path to Qwen3-TTS-12Hz-1.7B-VoiceDesign")
    ap.add_argument("--base", help="path to Qwen3-TTS-12Hz-1.7B-Base (consistent designed voices)")
    ap.add_argument("--num_designed", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_sentences", type=int, default=0,
                    help="sentences per speaker (0 = all)")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)

    with open(args.sentences, encoding="utf-8") as f:
        sentences = [s.strip() for s in f if s.strip()]
    if args.max_sentences:
        sentences = sentences[:args.max_sentences]
    print(f"{len(sentences)} sentences")

    os.makedirs(args.out, exist_ok=True)
    writer = Writer(args.out)

    # ---------------------------------------------------- preset speakers ---
    if args.custom:
        model = load(args.custom)
        speakers = PRESET_SPEAKERS or list(model.get_supported_speakers())
        print("preset speakers:", speakers)
        for spk in speakers:
            jobs = [(i, t, rng.choice(PRESET_STYLES)) for i, t in writer.todo(spk, sentences)]

            def gen(texts, styles, spk=spk):
                return model.generate_custom_voice(
                    text=texts, language=[args.language] * len(texts),
                    speaker=[spk] * len(texts), instruct=styles)

            run_batches(writer, spk, jobs, args.batch_size, gen)
        del model
        torch.cuda.empty_cache()

    # ---------------------------------------------------- designed voices ---
    if args.design:
        voices_path = os.path.join(args.out, "voices.json")
        if os.path.exists(voices_path):  # keep the same voices when resuming
            with open(voices_path, encoding="utf-8") as f:
                voices = json.load(f)
        else:
            voices = {}
        for k in range(len(voices), args.num_designed):
            voices[f"designed_{k:03d}"] = random_voice(rng)
        with open(voices_path, "w", encoding="utf-8") as f:
            json.dump(voices, f, indent=2, ensure_ascii=False)

        design = load(args.design)

        if args.base:
            # 1) design each voice once -> reference clip
            ref_dir = os.path.join(args.out, "voice_refs")
            os.makedirs(ref_dir, exist_ok=True)
            names = list(voices)
            missing = [n for n in names if not os.path.exists(os.path.join(ref_dir, n + ".wav"))]
            for batch in chunks(missing, args.batch_size):
                wavs, sr = design.generate_voice_design(
                    text=[REF_TEXT] * len(batch), language=[args.language] * len(batch),
                    instruct=[voices[n] for n in batch])
                for n, w in zip(batch, wavs):
                    sf.write(os.path.join(ref_dir, n + ".wav"), w, sr)
                    print("designed", n, "-", voices[n])
            del design
            torch.cuda.empty_cache()

            # 2) clone each reference for every sentence
            base = load(args.base)
            for n in names:
                jobs = [(i, t, voices[n]) for i, t in writer.todo(n, sentences)]
                if not jobs:
                    continue
                prompt = base.create_voice_clone_prompt(
                    ref_audio=os.path.join(ref_dir, n + ".wav"), ref_text=REF_TEXT)

                def gen(texts, styles, prompt=prompt):
                    return base.generate_voice_clone(
                        text=texts, language=[args.language] * len(texts),
                        voice_clone_prompt=prompt)

                run_batches(writer, n, jobs, args.batch_size, gen)
        else:
            # no Base model: design every line directly (speaker may drift)
            for n, desc in voices.items():
                jobs = [(i, t, desc) for i, t in writer.todo(n, sentences)]

                def gen(texts, styles):
                    return design.generate_voice_design(
                        text=texts, language=[args.language] * len(texts), instruct=styles)

                run_batches(writer, n, jobs, args.batch_size, gen)

    print("done ->", args.out)


if __name__ == "__main__":
    main()
