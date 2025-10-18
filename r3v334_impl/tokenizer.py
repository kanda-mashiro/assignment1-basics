class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens

    def encode(self, s: str) -> list[int]:
        pass

    def decode(self, tokens: list[int]) -> str:
        # 转 bytes
        bytes_list: list[bytes] = []
        for token in tokens:
            bytes_list.append(self.vocab[token])

        