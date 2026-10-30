"""Download every model the API uses at build time, reading names from config.yaml so they match runtime."""
import yaml

cfg = yaml.safe_load(open("config.yaml"))

import whisperx  # noqa: E402

name = cfg["asr"]["model"].split("/")[-1].removeprefix("whisper-")
print("ASR:", name, flush=True)
whisperx.load_model(name, "cpu", compute_type="int8", language=cfg["asr"]["language"])

print("alignment:", cfg["alignment"]["model"], flush=True)
whisperx.load_align_model(language_code="en", device="cpu", model_name=cfg["alignment"]["model"])

import nltk  # noqa: E402  (whisperx.align's sentence splitter)

nltk.download("punkt_tab", quiet=True)

print("spaCy:", cfg["detection"]["spacy_model"], flush=True)
import spacy  # noqa: E402

spacy.load(cfg["detection"]["spacy_model"])

print("BERTScore:", cfg["evaluation"]["bertscore_model"], flush=True)
from transformers import AutoModel, AutoTokenizer  # noqa: E402

AutoTokenizer.from_pretrained(cfg["evaluation"]["bertscore_model"])
AutoModel.from_pretrained(cfg["evaluation"]["bertscore_model"])
print("all models cached", flush=True)
