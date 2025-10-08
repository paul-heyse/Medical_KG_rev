from types import SimpleNamespace

import pytest

from Medical_KG_rev.services.chunking import sentence_splitters
from Medical_KG_rev.services.chunking.benchmark_sentence_segmenters import (
    benchmark_segmenters,
)
from Medical_KG_rev.services.chunking.wrappers import huggingface_segmenter as hf
from Medical_KG_rev.services.chunking.wrappers.huggingface_segmenter import (
    HuggingFaceSentenceSegmenter,
)
from Medical_KG_rev.services.chunking.wrappers.syntok_segmenter import (
    SyntokSentenceSegmenter,
)


@pytest.fixture(autouse=True)
def _restore_env(monkeypatch):
    monkeypatch.delenv("MEDICAL_KG_SENTENCE_MODEL", raising=False)
    yield

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


def test_default_loader_without_model_warns(monkeypatch):
    with pytest.warns(RuntimeWarning):
        loader = hf._default_loader()
    assert isinstance(loader, hf._HeuristicSentenceSplitter)
    assert loader("Single sentence") == [(0, 15, "Single sentence")]


def test_default_segmenter_invokes_default_loader(monkeypatch):
    called: list[str] = []

    def fake_loader():
        called.append("loader")

        def _segment(_: str):
            return []

        return _segment

    monkeypatch.setattr(hf, "_default_loader", fake_loader)
    segmenter = hf.default_segmenter()
    assert segmenter("text") == []
    assert called == ["loader"]


def test_tokenizer_sentence_splitter_handles_abbreviations_and_newlines():
    class FakeBackend:
        def __init__(self, tokens):
            self._tokens = tokens

        def pre_tokenize_str(self, _: str):
            return list(self._tokens)

    tokenizer = SimpleNamespace(
        backend_tokenizer=SimpleNamespace(
            pre_tokenizer=FakeBackend(
                [
                    ("Dr.", (0, 3)),
                    ("Alpha", (4, 9)),
                    ("Beta.", (10, 15)),
                    ("Gamma", (17, 22)),
                ]
            )
        )
    )

    splitter = hf._TokenizerSentenceSplitter(tokenizer)
    text = "Dr. Alpha Beta.\n\nGamma"
    segments = splitter(text)
    assert segments == [
        (0, 15, "Dr. Alpha Beta."),
        (17, 22, "Gamma"),
    ]


def test_tokenizer_sentence_splitter_breaks_on_double_newline():
    class FakeBackend:
        def pre_tokenize_str(self, _: str):
            return [
                ("Alpha", (0, 5)),
                ("\n\n", (5, 7)),
                ("Beta", (7, 11)),
            ]

    tokenizer = SimpleNamespace(
        backend_tokenizer=SimpleNamespace(pre_tokenizer=FakeBackend())
    )

    splitter = hf._TokenizerSentenceSplitter(tokenizer)
    text = "Alpha\n\nBeta"
    assert splitter(text) == [(0, 5, "Alpha"), (7, 11, "Beta")]


def test_tokenizer_sentence_splitter_requires_fast_backend():
    splitter = hf._TokenizerSentenceSplitter(SimpleNamespace(backend_tokenizer=None))
    with pytest.raises(RuntimeError):
        splitter("Hello")


def test_tokenizer_sentence_splitter_skips_whitespace_segments():
    class FakeBackend:
        def pre_tokenize_str(self, _: str):
            return [("   ", (0, 3))]

    tokenizer = SimpleNamespace(
        backend_tokenizer=SimpleNamespace(pre_tokenizer=FakeBackend())
    )

    splitter = hf._TokenizerSentenceSplitter(tokenizer)
    assert splitter("   ") == []


def test_tokenizer_sentence_splitter_returns_empty_when_no_tokens():
    class FakeBackend:
        def pre_tokenize_str(self, _: str):
            return []

    tokenizer = SimpleNamespace(
        backend_tokenizer=SimpleNamespace(pre_tokenizer=FakeBackend())
    )

    splitter = hf._TokenizerSentenceSplitter(tokenizer)
    assert splitter("Anything") == []


def test_tokenizer_sentence_splitter_respects_special_tokens():
    class FakeBackend:
        def pre_tokenize_str(self, _: str):
            return [
                ("Alpha", (0, 5)),
                ("</s>", (5, 9)),
                ("Omega", (9, 14)),
            ]

    tokenizer = SimpleNamespace(
        backend_tokenizer=SimpleNamespace(pre_tokenizer=FakeBackend())
    )

    splitter = hf._TokenizerSentenceSplitter(tokenizer)
    text = "Alpha</s>Omega"
    segments = splitter(text)
    assert segments == [
        (0, 9, "Alpha</s>"),
        (9, 14, "Omega"),
    ]


def test_heuristic_splitter_handles_whitespace():
    splitter = hf._HeuristicSentenceSplitter()
    assert splitter("  Leading and trailing  ") == [
        (2, 22, "Leading and trailing")
    ]


def test_heuristic_splitter_handles_missing_segments():
    class StrangeStr(str):
        def split(self, sep=None, maxsplit=-1):
            if sep == ". ":
                return ["   "]
            return super().split(sep, maxsplit)

    splitter = hf._HeuristicSentenceSplitter()
    text = StrangeStr("Alpha")
    assert splitter(text) == [(0, len(text), "Alpha")]


def test_heuristic_splitter_handles_missing_find():
    class WeirdStr(str):
        def find(self, sub, start=0, end=None):
            if start > 0:
                return -1
            return super().find(sub, start, len(self) if end is None else end)

    splitter = hf._HeuristicSentenceSplitter()
    text = WeirdStr("Alpha. Beta.")
    segments = splitter(text)
    assert segments[0] == (0, 5, "Alpha")
    assert segments[1][0] == 5
    assert segments[1][2].startswith(". ")


def test_should_close_sentence_handles_empty_and_newline():
    assert hf._should_close_sentence("   ", "text", 0, 1) is False
    assert hf._should_close_sentence("Alpha", "Alpha\n\nBeta", 0, 5) is True


def test_trim_offsets_trims_both_sides():
    assert hf._trim_offsets("  text  ", 0, 7) == (2, 6)


def test_merge_abbreviation_segments_with_empty_input():
    assert hf._merge_abbreviation_segments("text", []) == []


def test_get_sentence_splitter_defaults_to_simple():
    splitter = sentence_splitters.get_sentence_splitter("unknown")
    assert splitter("Alpha. Beta.") == [(0, 5, "Alpha"), (7, 12, "Beta.")]


def test_get_sentence_splitter_warns_on_scispacy(monkeypatch):
    with pytest.warns(DeprecationWarning):
        splitter = sentence_splitters.get_sentence_splitter("scispacy")
    # Ensure the returned splitter is the huggingface path by stubbing segmenter
    monkeypatch.setattr(hf, "default_segmenter", lambda: lambda text: [(0, len(text), text)])
    assert splitter("Hello") == [(0, 5, "Hello")]


def test_get_sentence_splitter_syntok(monkeypatch):
    called: list[str] = []

    def fake_segmenter():
        called.append("syntok")

        class _Splitter:
            def segment(self, text: str):
                return [(0, len(text), text.upper())]

        return _Splitter()

    monkeypatch.setattr(sentence_splitters, "_syntok_segmenter", fake_segmenter)
    splitter = sentence_splitters.get_sentence_splitter("syntok")
    assert splitter("abc") == [(0, 3, "ABC")]
    assert called == ["syntok"]


def test_syntok_segmenter_offsets():
    def analyzer(_: str):
        yield [
            [
                SimpleNamespace(spacing="", value="Sentence"),
                SimpleNamespace(spacing=" ", value="one."),
            ],
            [
                SimpleNamespace(spacing="", value="Sentence"),
                SimpleNamespace(spacing=" ", value="two."),
            ],
        ]

    segmenter = SyntokSentenceSegmenter(analyzer_factory=lambda: analyzer)
    text = "Sentence one. Sentence two."
    segments = segmenter.segment(text)
    assert segments == [
        (0, 13, "Sentence one."),
        (14, 27, "Sentence two."),
    ]


def _timer(values):
    iterator = iter(values)

    def _now():
        return next(iterator)

    return _now


def test_benchmark_segmenters_reports_throughput():
    corpus = ["Alpha beta", "Gamma delta"]

    def hf_segmenter(text: str):
        return [(0, len(text), text)]

    def syntok_segmenter(text: str):
        midpoint = len(text) // 2
        return [
            (0, midpoint, text[:midpoint]),
            (midpoint, len(text), text[midpoint:]),
        ]

    timer = _timer([0.0, 0.2, 0.2, 0.5])
    results = benchmark_segmenters(
        {"hf": hf_segmenter, "syntok": syntok_segmenter},
        corpus,
        timer=timer,
    )

    assert [result.name for result in results] == ["hf", "syntok"]
    hf_result = next(result for result in results if result.name == "hf")
    assert hf_result.documents == 2
    assert hf_result.sentences == 2
    assert hf_result.duration_seconds == pytest.approx(0.2)
    assert hf_result.throughput_docs_per_second == pytest.approx(10.0)
    assert hf_result.throughput_sentences_per_second == pytest.approx(10.0)


def test_benchmark_segmenters_requires_positive_repeats():
    with pytest.raises(ValueError):
        benchmark_segmenters({"hf": lambda text: []}, ["text"], repeats=0)
