from Medical_KG_rev.services.chunking.wrappers import tokenizers


def test_hf_tokenizer_caching(monkeypatch):
    class FakeTokenizer:
        def __init__(self, model: str) -> None:
            self.model = model

        def encode(self, text: str, add_special_tokens: bool = False):
            return list(range(len(text.split())))

    def fake_from_pretrained(model_name: str):
        return FakeTokenizer(model_name)

    monkeypatch.setattr(
        tokenizers, "_load_hf_tokenizer", lambda model_name: fake_from_pretrained(model_name)
    )
    tokenizers.get_hf_tokenizer.cache_clear()
    value = tokenizers.count_tokens_hf("hello world", model_name="fake-model")
    assert value == 2
    value = tokenizers.count_tokens_hf("hello", model_name="fake-model")
    assert value == 1
    tokenizer_instance = tokenizers.get_hf_tokenizer("fake-model")
    assert tokenizer_instance is tokenizers.get_hf_tokenizer("fake-model")


def test_tiktoken_budget(monkeypatch):
    class FakeEncoder:
        def encode(self, text: str):
            return list(text)

    monkeypatch.setattr(tokenizers, "_load_tiktoken", lambda model_name: FakeEncoder())
    tokenizers.get_tiktoken_encoder.cache_clear()
    tokens = tokenizers.count_tokens_tiktoken("abc")
    assert tokens == 3
    assert tokenizers.ensure_within_budget(
        "abcd", budget=5, counter=tokenizers.count_tokens_tiktoken
    )
