"""Shared emotion definitions for the Qwen3-TTS scripts.

Each emotion has
  instruct : instruction for CustomVoice (preset speakers)
  design   : phrase appended to a VoiceDesign voice description
  ref_text : sentence used to build an emotional reference clip for Base cloning
             (the clone copies the manner of its reference, so the text should suit the emotion)
"""

EMOTIONS = {
    "neutral": {
        "instruct": "",
        "design": "speaking in a neutral, matter-of-fact way",
        "ref_text": "Hello, it is nice to meet you. Today I will read a few sentences for you in my natural speaking voice.",
    },
    "happy": {
        "instruct": "Speak happily, in a bright and cheerful tone.",
        "design": "sounding happy and cheerful, smiling while speaking",
        "ref_text": "Oh, this is wonderful news! I am so happy today, everything is going really well.",
    },
    "angry": {
        "instruct": "Speak angrily, with clear irritation and frustration.",
        "design": "sounding angry and frustrated, speaking forcefully",
        "ref_text": "I have asked you three times already! This is not acceptable, just do it properly.",
    },
    "sad": {
        "instruct": "Speak sadly, in a low, quiet and downcast voice.",
        "design": "sounding sad and downcast, speaking quietly",
        "ref_text": "I am not feeling good today. Everything seems so heavy, and I just miss how things used to be.",
    },
    "fearful": {
        "instruct": "Speak anxiously and fearfully, with urgency and a shaky voice.",
        "design": "sounding anxious and frightened, speaking with urgency and a shaky voice",
        "ref_text": "Please, someone help, something is wrong! I am really scared, we need to hurry.",
    },
    "tired": {
        "instruct": "Speak in a tired, sleepy, low-energy voice.",
        "design": "sounding tired and sleepy, with low energy",
        "ref_text": "It has been such a long day. I am so tired, I can barely keep my eyes open.",
    },
    "calm": {
        "instruct": "Speak calmly, slowly and gently.",
        "design": "sounding calm and relaxed, speaking slowly and gently",
        "ref_text": "Take a deep breath. There is no rush, everything is fine, and we can take our time.",
    },
}

# Intents where a matching emotion is more realistic; used more often for those intents.
INTENT_EMOTION_BIAS = {
    "healthcare_alert": ["fearful", "fearful", "sad"],
    "healthcare_assist": ["tired", "calm"],
    "navigation_stop": ["angry", "fearful"],
}
