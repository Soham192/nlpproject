"""Synthesize a mock support call with planted (fake) sensitive entities, plus exact ground truth.

Each entity is synthesized as its own TTS chunk, so its start/end time in the
final WAV is known exactly — that is the hand-label ground truth used for
entity precision/recall. Requires internet (gTTS) and ffmpeg.

    python scripts/make_sample.py                        # -> data/samples/mock_call.wav + .ground_truth.json
    python scripts/make_sample.py --call meridian_call   # -> data/samples/meridian_call.wav + ...
"""
from __future__ import annotations

import argparse
import io
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.audio import AudioParams, write_wav  # noqa: E402

SR = 16000
TURN_GAP = 0.6
CHUNK_GAP = 0.08

AGENT, CUSTOMER = "com", "co.uk"  # gTTS accents as two speakers
PRIYA, ROHAN = "co.in", "com.au"

# (speaker, [(text, entity_type | None), ...]) — all values are fake.
# Card 4111 1111 1111 1111 is the standard Luhn-valid Visa test number.
MOCK_CALL = [
    (AGENT, [("Thank you for calling First Harbor support. Can I get your full name please?", None)]),
    (CUSTOMER, [("Sure, my name is", None), ("Michael Johnson", "PERSON"), (".", None)]),
    (AGENT, [("Thanks. And your date of birth?", None)]),
    (CUSTOMER, [("It's", None), ("March 14th, 1985", "DATE_TIME"), (".", None)]),
    (AGENT, [("Great. What can I help you with today?", None)]),
    (CUSTOMER, [("I was charged twice for my last order and I'd like a refund to my card.", None)]),
    (AGENT, [("No problem. Can you read me the card number?", None)]),
    (CUSTOMER, [("Yes, it's", None), ("4 1 1 1, 1 1 1 1, 1 1 1 1, 1 1 1 1", "CREDIT_CARD"), (".", None)]),
    (AGENT, [("And for verification, the last digits of your social security number, or the full number if you prefer.", None)]),
    (CUSTOMER, [("My social security number is", None), ("5 1 2, 3 8, 4 7 2 1", "US_SSN"), (".", None)]),
    (AGENT, [("Thank you. What's the best phone number to reach you?", None)]),
    (CUSTOMER, [("You can call me at", None), ("4 1 5, 5 5 5, 0 1 3 2", "PHONE_NUMBER"), (".", None)]),
    (AGENT, [("And you're still at the address in", None), ("Springfield, Illinois", "LOCATION"), ("?", None)]),
    (CUSTOMER, [("Yes, that's right.", None)]),
    (AGENT, [("Perfect. I've issued the refund. It should appear within five business days. "
              "Is there anything else I can help with?", None)]),
    (CUSTOMER, [("No, that's everything. Thanks for your help.", None)]),
]

# User-supplied bank-dispute script. Digits are spaced so TTS reads them
# digit-by-digit the way a caller would; groupings follow natural speech.
MERIDIAN_CALL = [
    (PRIYA, [("Thanks for calling Meridian Bank. This is", None), ("Priya", "PERSON"),
             ("speaking. How can I help you today?", None)]),
    (ROHAN, [("Hi", None), ("Priya", "PERSON"), (", my name is", None), ("Rohan Mehta", "PERSON"),
             ("and I'm calling about a charge on my statement I don't recognize.", None)]),
    (PRIYA, [("I can certainly help with that,", None), ("Rohan", "PERSON"),
             (". Can I start with your account number please?", None)]),
    (ROHAN, [("Yes, it's", None), ("8 8 2 4, 1 9 0 3", "ACCOUNT_NUMBER"),
             ("and the card is the Visa ending 1 1 1.", None)]),
    (PRIYA, [("Could you read me the full card number?", None)]),
    (ROHAN, [("Sure.", None), ("4 1 1 1, 1 1 1 1, 1 1 1 1, 1 1 1 1", "CREDIT_CARD"), (".", None)]),
    (PRIYA, [("Thank you. For verification, can you confirm your date of birth?", None)]),
    (ROHAN, [("It's the", None), ("14th of March, 1988", "DATE_TIME"), (".", None)]),
    (PRIYA, [("And your social security number?", None)]),
    (ROHAN, [("It's", None), ("9 0 0, 5 5, 4 8 2 1", "US_SSN"), (".", None)]),
    (PRIYA, [("Perfect. That all matches. What's the best number to reach you on?", None)]),
    (ROHAN, [("5 5 5, 0 1 4 7", "PHONE_NUMBER"), (".", None)]),
    (PRIYA, [("And an email address for the confirmation?", None)]),
    (ROHAN, [("It's", None), ("rohan.mehta@example.com", "EMAIL_ADDRESS"), (".", None)]),
    (PRIYA, [("Got it. Now tell me about the charge.", None)]),
    (ROHAN, [("There's a charge for 12,000 rupees from a shop in", None), ("Pune", "LOCATION"),
             ("that I've never been to in my life.", None)]),
    (PRIYA, [("I can see it here. That one posted today.", None)]),
    (ROHAN, [("Right. And I was in", None), ("Nagpur", "LOCATION"),
             ("the whole week, so it definitely wasn't mine.", None)]),
    (PRIYA, [("Understood. I'll raise a dispute on it now. "
              "You should see a provisional credit within 5 business days.", None)]),
    (ROHAN, [("That's great. Do you need my policy number for travel cover as well?", None)]),
    (PRIYA, [("If you have it handy, yes please.", None)]),
    (ROHAN, [("It's", None), ("P 4 7 7 3 1", "POLICY_NUMBER"), (".", None)]),
    (PRIYA, [("Thanks", None), ("Rohan", "PERSON"), (", that's everything on my side. Anything else I can help with?", None)]),
    (ROHAN, [("No, that covers it. Thanks for your help,", None), ("Priya", "PERSON"), (".", None)]),
]

CALLS = {"mock_call": MOCK_CALL, "meridian_call": MERIDIAN_CALL}


def tts(text: str, tld: str) -> np.ndarray:
    from gtts import gTTS

    buf = io.BytesIO()
    gTTS(text=text, lang="en", tld=tld).write_to_fp(buf)
    pcm = subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-i", "pipe:0", "-ac", "1", "-ar", str(SR), "-f", "s16le", "pipe:1"],
        input=buf.getvalue(), capture_output=True, check=True).stdout
    return np.frombuffer(pcm, dtype="<i2")


def trim(x: np.ndarray, thresh: int = 300) -> np.ndarray:
    idx = np.flatnonzero(np.abs(x.astype(np.int32)) > thresh)
    return x[idx[0]:idx[-1] + 1] if len(idx) else x[:0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--call", choices=sorted(CALLS), default="mock_call")
    ap.add_argument("-o", "--out", help="default: data/samples/<call>.wav")
    args = ap.parse_args()
    call = CALLS[args.call]
    args.out = args.out or str(ROOT / f"data/samples/{args.call}.wav")

    pieces: list[np.ndarray] = [np.zeros(int(0.3 * SR), np.int16)]
    pos = len(pieces[0])
    truth = []
    for t_idx, (tld, chunks) in enumerate(call):
        if t_idx:
            pieces.append(np.zeros(int(TURN_GAP * SR), np.int16))
            pos += len(pieces[-1])
        for c_idx, (text, ent) in enumerate(chunks):
            if not any(ch.isalnum() for ch in text):
                continue
            if c_idx:
                pieces.append(np.zeros(int(CHUNK_GAP * SR), np.int16))
                pos += len(pieces[-1])
            audio = trim(tts(text, tld))
            if ent:
                truth.append({"entity_type": ent, "text": text, "start_sec": round(pos / SR, 3),
                              "end_sec": round((pos + len(audio)) / SR, 3)})
            pieces.append(audio)
            pos += len(audio)
            print(f"  {pos / SR:6.2f}s  {'[' + ent + '] ' if ent else ''}{text}")
    pieces.append(np.zeros(int(0.5 * SR), np.int16))
    samples = np.concatenate(pieces)[:, None]

    out = Path(args.out)
    write_wav(out, samples, AudioParams(SR, 1, 2, len(samples)))
    gt = out.with_name(out.stem + ".ground_truth.json")
    gt.write_text(json.dumps({"audio": out.name, "entities": truth}, indent=2))
    print(f"wrote {out} ({len(samples) / SR:.1f}s) and {gt} ({len(truth)} entities)")


if __name__ == "__main__":
    main()
