import regex

from collections.abc import Iterable, Iterator
from cs336_basics import PAT

class Tokenizer:
    def __init__(self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = vocab
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        self.merges = merges
        self.special_tokens = special_tokens
        if self.special_tokens:
            # 按照长度排序，可处理 special token 出现 overlap 的情况
            self.special_tokens.sort(key=len, reverse=True)


    def _pretokenization(self, s: str) -> list[str]:
        chunks: list[str] = []
        if self.special_tokens:
            patten = "|".join([regex.escape(token) for token in self.special_tokens])
            patten = "(" + patten + ")"
            for chunk in regex.split(patten, s):
                if len(chunk) == 0:
                    continue
                if chunk in self.special_tokens:
                    chunks.append(chunk)
                else:
                    chunks.extend(regex.findall(PAT, chunk))
        else:
            chunks.extend(regex.findall(PAT, s))

        return chunks

    def _encode(self, s: str) -> list[int]:
        ids: list[int] = []
        if self.special_tokens and s in self.special_tokens:
            ids.append(self.reverse_vocab[s.encode("utf-8")])
        else:
            bs = [bytes([b]) for b in s.encode('utf-8')]
            for t1, t2 in self.merges:
                new_bs: list[bytes] = []
                idx = 0
                while idx < len(bs):
                    if idx + 1 < len(bs) and bs[idx] == t1 and bs[idx+1] == t2:
                        new_bs.append(t1 + t2)
                        idx += 2
                    else:
                        new_bs.append(bs[idx])
                        idx += 1
                bs = new_bs
            for x in bs:
                ids.append(self.reverse_vocab[x])

        return ids


    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        # 每次拿到一行数据
        for line in iterable:
            for chunk in self._pretokenization(line):
                for id in self._encode(chunk):
                    yield id

    def encode(self, s: str) -> list[int]:
        ids: list[int] = []
        for chunk in self._pretokenization(s):
            ids.extend(self._encode(chunk))
        return ids

    def decode(self, ids: list[int]) -> str:
        return b"".join([self.vocab[id] for id in ids]).decode("utf-8", errors="replace")