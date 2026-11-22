"""
Preview Qwen3-TTS emotions before generating the full dataset.

Makes the same few robot commands in every emotion with three methods:
  A. CustomVoice preset speakers + emotion instruction        (needs --custom)
  B. VoiceDesign: voice description + emotion, per line        (needs --design)
  C. VoiceDesign emotional reference clip -> Base voice clone  (needs --design and --base)
and writes emotion_preview/index.html, a page with an audio player for every sample.

Example:
  export HF_HUB_OFFLINE=1
  python scripts/emotion_preview_qwen3.py \
      --custom ./Qwen3-TTS-12Hz-1.7B-CustomVoice \
      --design ./Qwen3-TTS-12Hz-1.7B-VoiceDesign \
      --base   ./Qwen3-TTS-12Hz-1.7B-Base
  # then open emotion_preview/index.html in a browser
"""
import argparse
import html
import os

import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel

from emotions import EMOTIONS

SENTENCES = [
    "Stop right there.",
    "Can you switch off the A C? I am feeling cold.",
    "Call for help, I have fallen.",
    "Bring me a glass of water.",
]
PRESET = ["Ryan", "Aiden"]  # English preset speakers
VOICE = "An adult in their thirties female voice, medium-pitched and warm, with a neutral American accent"


def load(path):
    return Qwen3TTSModel.from_pretrained(path, device_map="cuda:0", dtype=torch.bfloat16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--custom")
    ap.add_argument("--design")
    ap.add_argument("--base")
    ap.add_argument("--out", default="emotion_preview")
    ap.add_argument("--language", default="English")
    ap.add_argument("--emotions", nargs="*", default=list(EMOTIONS))
    ap.add_argument("--voice", default=VOICE, help="voice description for methods B and C")
    ap.add_argument("--seed", type=int, default=1234)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    emos, n = a.emotions, len(SENTENCES)
    rows = []  # (method, speaker, emotion, sentence_idx, relpath)

    def save(method, spk, emo, i, w, sr):
        rel = f"{method}/{spk}/s{i + 1}_{emo}.wav"
        os.makedirs(os.path.join(a.out, os.path.dirname(rel)), exist_ok=True)
        sf.write(os.path.join(a.out, rel), w, sr)
        rows.append((method, spk, emo, i, rel))
        print("ok", rel)

    if a.custom:
        m = load(a.custom)
        for spk in PRESET:
            for emo in emos:
                wavs, sr = m.generate_custom_voice(
                    text=SENTENCES, language=[a.language] * n, speaker=[spk] * n,
                    instruct=[EMOTIONS[emo]["instruct"]] * n)
                for i, w in enumerate(wavs):
                    save("A_customvoice", spk, emo, i, w, sr)
        del m
        torch.cuda.empty_cache()

    refs = {}
    if a.design:
        m = load(a.design)
        for emo in emos:
            desc = f"{a.voice}, {EMOTIONS[emo]['design']}."
            wavs, sr = m.generate_voice_design(text=SENTENCES, language=[a.language] * n,
                                               instruct=[desc] * n)
            for i, w in enumerate(wavs):
                save("B_voicedesign", "designed", emo, i, w, sr)
            if a.base:  # emotional reference clip for method C
                wv, sr = m.generate_voice_design(text=[EMOTIONS[emo]["ref_text"]],
                                                 language=[a.language], instruct=[desc])
                p = os.path.join(a.out, "C_clone/refs", f"ref_{emo}.wav")
                os.makedirs(os.path.dirname(p), exist_ok=True)
                sf.write(p, wv[0], sr)
                refs[emo] = p
                rows.append(("C_clone", "reference", emo, -1, os.path.relpath(p, a.out)))
        del m
        torch.cuda.empty_cache()

    if a.base and refs:
        m = load(a.base)
        for emo, p in refs.items():
            prompt = m.create_voice_clone_prompt(ref_audio=p, ref_text=EMOTIONS[emo]["ref_text"])
            wavs, sr = m.generate_voice_clone(text=SENTENCES, language=[a.language] * n,
                                              voice_clone_prompt=prompt)
            for i, w in enumerate(wavs):
                save("C_clone", "cloned", emo, i, w, sr)

    # ------------------------------------------------------------ HTML page ---
    groups = {}
    for method, spk, emo, i, rel in rows:
        groups.setdefault((method, spk), {})[(emo, i)] = rel
    titles = {"A_customvoice": "A. CustomVoice preset speaker + emotion instruction",
              "B_voicedesign": "B. VoiceDesign, emotion in the description (per line)",
              "C_clone": "C. Base clone of an emotional reference clip"}
    out = ["<!DOCTYPE html><html><head><meta charset='utf-8'><title>Qwen3-TTS emotion preview</title>",
           "<style>body{font-family:system-ui,sans-serif;margin:24px;color:#222}"
           "table{border-collapse:collapse;margin-bottom:32px}td,th{border:1px solid #ddd;padding:6px;"
           "vertical-align:top;font-size:13px}th{background:#f4f4f4}audio{width:190px;height:32px}</style>",
           "</head><body><h1>Qwen3-TTS emotion preview</h1>",
           f"<p>Voice for B/C: {html.escape(a.voice)}</p>"]
    for (method, spk), cells in groups.items():
        out.append(f"<h2>{titles[method]} - {html.escape(spk)}</h2><table><tr><th>emotion</th>")
        idxs = sorted({i for (_, i) in cells})
        for i in idxs:
            label = "reference clip" if i < 0 else SENTENCES[i]
            out.append(f"<th>{html.escape(label)}</th>")
        out.append("</tr>")
        for emo in emos:
            out.append(f"<tr><td><b>{emo}</b><br><small>{html.escape(EMOTIONS[emo]['instruct'] or 'no instruction')}</small></td>")
            for i in idxs:
                rel = cells.get((emo, i))
                out.append(f"<td><audio controls preload='none' src='{rel}'></audio></td>" if rel else "<td></td>")
            out.append("</tr>")
        out.append("</table>")
    out.append("</body></html>")
    with open(os.path.join(a.out, "index.html"), "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    print(f"{len(rows)} clips -> open {os.path.join(a.out, 'index.html')}")


if __name__ == "__main__":
    main()
