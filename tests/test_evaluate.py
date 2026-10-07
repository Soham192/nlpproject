from src.config import load_config
from src.evaluate import entity_metrics, normalize_text, unredacted_text
from src.alignment import Word
from tests.test_manifest import _manifest, _span


def test_normalization_matches_on_both_sides():
    n = load_config().evaluation.normalize
    assert normalize_text("Don't STOP, it's fine!", n) == normalize_text("do not stop it is fine", n)


def test_unredacted_text_drops_masked_words():
    words = [Word("a", 0, 0.5), Word("secret", 1.0, 1.5), Word("b", 2.0, 2.5)]
    assert unredacted_text(words, [(0.9, 1.6)]) == "a b"


def test_entity_metrics():
    m = _manifest([_span(1, 0, 16000 // 2)])  # masks 0-0.5 s
    gt = [{"entity_type": "CREDIT_CARD", "start_sec": 0.1, "end_sec": 0.4},
          {"entity_type": "PERSON", "start_sec": 0.7, "end_sec": 0.9}]
    r = entity_metrics(m, gt)
    assert r["precision"] == 1.0 and r["recall"] == 0.5
    assert r["by_type"]["PERSON"]["recall"] == 0.0
