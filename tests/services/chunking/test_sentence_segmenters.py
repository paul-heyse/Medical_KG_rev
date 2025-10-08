from dataclasses import dataclass
from types import SimpleNamespace

from Medical_KG_rev.services.chunking.wrappers.scispacy_segmenter import (
    SciSpaCySentenceSegmenter,
)
from Medical_KG_rev.services.chunking.wrappers.syntok_segmenter import (
    SyntokSentenceSegmenter,
)


@dataclass
class _FakeSpan:
    start_char: int
    end_char: int


def test_scispacy_segmenter_offsets():
    def loader():
        class _Model:
            def __call__(self, text: str):
                spans = [
                    _FakeSpan(0, 13),
                    _FakeSpan(14, len(text)),
                ]
                return SimpleNamespace(sents=spans)

        return _Model()

    segmenter = SciSpaCySentenceSegmenter(loader=lambda: loader())
    text = "Sentence one. Sentence two"
    segments = segmenter.segment(text)
    assert segments == [
        (0, 13, "Sentence one."),
        (14, len(text), "Sentence two"),
    ]


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
