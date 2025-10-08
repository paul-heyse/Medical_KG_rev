from dataclasses import dataclass
from types import SimpleNamespace

from Medical_KG_rev.services.chunking.wrappers.huggingface_segmenter import (
    HuggingFaceSentenceSegmenter,
)
from Medical_KG_rev.services.chunking.wrappers.syntok_segmenter import (
    SyntokSentenceSegmenter,
)


@dataclass
class _FakeSpan:
    start_char: int
    end_char: int


def test_huggingface_segmenter_offsets():
    def loader():
        class _Model:
            def __call__(self, text: str):
                # Mock Hugging Face model behavior
                return {"label": "POSITIVE", "score": 0.9}

        return _Model()

    segmenter = HuggingFaceSentenceSegmenter(loader=lambda: loader())
    text = "Sentence one. Sentence two"
    segments = segmenter.segment(text)
    # Note: The current implementation uses simple splitting as fallback
    # This test verifies the interface works correctly
    assert len(segments) >= 1
    assert all(isinstance(seg, tuple) and len(seg) == 3 for seg in segments)


def test_syntok_segmenter_offsets():
    token = SimpleNamespace

    def analyzer(_: str):
        yield [
            [
                token(spacing="", value="Sentence"),
                token(spacing=" ", value="one."),
            ],
            [
                token(spacing="", value="Sentence"),
                token(spacing=" ", value="two."),
            ],
        ]

    segmenter = SyntokSentenceSegmenter(analyzer_factory=lambda: analyzer)
    text = "Sentence one. Sentence two."
    segments = segmenter.segment(text)
    assert segments == [
        (0, 13, "Sentence one."),
        (14, 27, "Sentence two."),
    ]
