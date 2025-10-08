from types import SimpleNamespace

from Medical_KG_rev.services.chunking.wrappers.huggingface_segmenter import (
    HuggingFaceSentenceSegmenter,
)
from Medical_KG_rev.services.chunking.wrappers.syntok_segmenter import (
    SyntokSentenceSegmenter,
)

def test_huggingface_segmenter_offsets():
    def loader():
        def _segment(text: str):
            return [
                (0, 13, "Sentence one."),
                (14, len(text), "Sentence two"),
            ]

        return _segment

    segmenter = HuggingFaceSentenceSegmenter(loader=lambda: loader())
    text = "Sentence one. Sentence two"
    segments = segmenter.segment(text)
    assert segments == [
        (0, 13, "Sentence one."),
        (14, len(text), "Sentence two"),
    ]


def test_huggingface_segmenter_merges_abbreviations():
    text = "Fig. 1 shows reduced error."

    def loader():
        def _segment(_: str):
            return [
                (0, 5, text[0:5]),
                (5, len(text), text[5:]),
            ]

        return _segment

    segmenter = HuggingFaceSentenceSegmenter(loader=lambda: loader())
    segments = segmenter.segment(text)
    assert segments == [(0, len(text), text)]


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
