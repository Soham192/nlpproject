"""Test priority 5: spoken-number normalisation and detection."""
import pytest

from src.alignment import Word, spell_digits
from src.detect import build_transcript


def W(text):
    return [Word(w, i * 0.3, i * 0.3 + 0.25) for i, w in enumerate(text.split())]


@pytest.mark.parametrize("spoken,expected", [
    ("four five three two zero one five one one two eight three zero three six six", "4532015112830366"),
    ("forty-five thirty-two oh one fifty-one", "45320151"),
    ("double five five one two three", "555123"),
    ("four hundred fifty two", "452"),
    ("four five three two, zero one five one, one two eight three, zero three six six", "4532-0151-1283-0366"),
    ("4532 0151 1283 0366", "4532-0151-1283-0366"),
    ("4-1-1-1, 1-1-1-1, 1-1-1-1, 1-1-1-1.", "4111-1111-1111-1111."),   # observed Whisper output
    ("5-1-2, 3-8, 4-7-2-1.", "512-38-4721."),
    ("one two three, four five, six seven eight nine.", "123-45-6789."),
])
def test_normalize_spoken_numbers(spoken, expected):
    assert build_transcript(W(spoken)).text == expected


def test_short_spoken_numbers_untouched():
    t = build_transcript(W("give me one moment, oh and two weeks"))
    assert t.text == "give me one moment, oh and two weeks"


def test_char_spans_map_back_to_words():
    words = W("my card is four five three two thanks")
    t = build_transcript(words)
    assert t.text == "my card is 4532 thanks"
    i = t.text.index("4532")
    hits = [k for k, (a, b) in enumerate(t.word_char_spans) if a < i + 4 and b > i]
    assert hits == [3, 4, 5, 6]


def test_normalization_off():
    assert build_transcript(W("four five three two"), normalize_numbers=False).text == "four five three two"


def test_empty_entity_selection_detects_nothing():
    # Presidio treats entities=[] as "all"; an empty UI selection must mask nothing instead.
    import dataclasses
    from src.config import load_config
    from src.detect import detect
    cfg = dataclasses.replace(load_config().detection, entities=())
    assert detect(build_transcript(W("my name is John Smith")), cfg) == []


def test_spell_digits_for_alignment():
    assert spell_digits("4532,") == ["four", "five", "three", "two,"]
    assert spell_digits("card") == ["card"]


presidio = pytest.importorskip("presidio_analyzer")


@pytest.fixture(scope="module")
def det_cfg():
    from src.config import load_config
    return load_config().detection


def _types(text, det_cfg):
    from src.detect import detect, split_by_threshold
    dets, _ = split_by_threshold(detect(build_transcript(W(text)), det_cfg), det_cfg.score_threshold)
    return {d.entity_type for d in dets}, dets


def test_spoken_card_detected(det_cfg):
    types, _ = _types("my card number is four five three two eight eight nine one "
                      "two three four five six seven eight seven", det_cfg)  # Luhn-valid
    assert "CREDIT_CARD" in types


def test_spoken_ssn_detected(det_cfg):
    types, _ = _types("my social security number is one two three, four five, six seven eight nine.", det_cfg)
    assert "US_SSN" in types


def test_person_detected(det_cfg):
    types, _ = _types("Hello, my name is John Smith and I live in Boston.", det_cfg)
    assert "PERSON" in types
