"""Test priority 4 + 6: overlap merging and segment-boundary entities."""
from src.alignment import Word
from src.detect import Detection, build_transcript
from src.mapping import SampleSpan, map_detections, merge_ranges, seconds_to_samples

SR = 16000


def test_entities_50ms_apart_merge_into_one_range():
    # Two entities 50 ms apart, 100 ms padding each side -> padded ranges overlap.
    words = [Word("John", 1.00, 1.40), Word("Smith", 1.45, 1.90)]
    t = build_transcript(words)
    dets = [Detection("PERSON", 0.85, 0, 4, "John"), Detection("LOCATION", 0.8, 5, 10, "Smith")]
    spans = map_detections(dets, t, SR, 10 * SR, padding_ms=100)
    assert len(spans) == 1
    s = spans[0]
    assert (s.sample_start, s.sample_end) == (int(0.9 * SR), int(2.0 * SR))
    assert s.entity_type == "PERSON+LOCATION"


def test_merge_sorted_non_overlapping():
    spans = merge_ranges([SampleSpan(500, 900), SampleSpan(0, 100), SampleSpan(880, 1000), SampleSpan(100, 200)])
    assert [(s.sample_start, s.sample_end) for s in spans] == [(0, 200), (500, 1000)]


def test_padding_clamped_to_file():
    assert seconds_to_samples(0.02, 0.5, SR, 100, 16000) == (0, int(0.6 * SR))
    assert seconds_to_samples(0.5, 0.99, SR, 100, 16000) == (int(0.4 * SR), 16000)


def test_entity_spanning_segment_boundary():
    # Whisper split "123-45-6789" across two segments; detection runs on the joined text.
    words = [Word("ssn", 0.0, 0.3), Word("one", 0.4, 0.6), Word("two", 0.6, 0.8), Word("three,", 0.8, 1.0),
             # --- segment boundary here ---
             Word("four", 1.2, 1.4), Word("five,", 1.4, 1.6), Word("six", 1.7, 1.9), Word("seven", 1.9, 2.1),
             Word("eight", 2.1, 2.3), Word("nine.", 2.3, 2.5)]
    t = build_transcript(words)
    assert "123-45-6789" in t.text
    i = t.text.index("123-45-6789")
    spans = map_detections([Detection("US_SSN", 0.85, i, i + 11, "123-45-6789")], t, SR, 3 * SR, 100)
    assert len(spans) == 1
    assert spans[0].sample_start == int(0.3 * SR) and spans[0].sample_end == int(2.6 * SR)
    assert (spans[0].word_start_idx, spans[0].word_end_idx) == (1, 10)
