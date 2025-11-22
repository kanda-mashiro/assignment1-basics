import itertools
import re
import token
from typing import Iterable
import regex as rex
    
def pre_tokenize(s: str, special_tokens: list[str]) -> list[str]:
    if special_tokens:
        p = "(" + "|".join([re.escape(special_token) for special_token in special_tokens]) + ")"
        chunks = []
        for item in re.split(p, s):
            if len(item) > 0:
                chunks.append(item)
    else:
        chunks = [s]

    print("cunks", chunks)

    
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    pretokens = []
    for chunk in chunks:
        if special_tokens:
            if chunk in special_tokens:
                pretokens.append(chunk)
                continue
        pretokens.extend([data for data in rex.findall(PAT, chunk)])

    return pretokens


def token_to_single_byte_list(token: str) -> list[bytes]:
    token_bytes: bytes = token.encode("utf-8")
    bytes_list = [
        token_bytes[idx:idx+1]
        for idx in range(len(token_bytes))
    ]

    return bytes_list

class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        # 长的在前面（认为需有优先匹配）
        # 贪婪匹配，认为越长的越应该去匹配
        if special_tokens:
            special_tokens.sort(
                key= lambda x: len(x),
                reverse=True
            )
        print("spt: ", special_tokens)

        # max_key = max(vocab.keys())
        # if special_tokens:
        #     for special_token in special_tokens:
        #         max_key += 1
        #         vocab[max_key] = special_token.encode("utf-8")
        #         print(max_key, vocab[max_key])

        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens

        # print(self.merges[0:10])

        # 可以 merge 加速
        self.merges_set: set[tuple[bytes, bytes]] = set(self.merges)
        # 加速 bytes
        self.reverse_vocab = {
            v: k
            for k, v in self.vocab.items()
        }

    def can_merge(self, pair: tuple[bytes, bytes]) -> int:
        for idx, value in enumerate(self.merges):
            if value == pair:
                return idx
        return -1


    def encode_pretoken(self, pretoken: str) -> list[int]:
        # import pdb
        # pdb.set_trace()

        if self.special_tokens and pretoken in self.special_tokens:
            return [self.reverse_vocab[pretoken.encode("utf-8")]]

        bytes_list = token_to_single_byte_list(pretoken)
        while True:
            merge_table = {}
            for idx in range(len(bytes_list)):
                if idx+1 >= len(bytes_list):
                    continue

                pair = (bytes_list[idx], bytes_list[idx+1])
                # print(pair, "can_merge: ", self.can_merge(pair))
                merge_idx = self.can_merge(pair)
                if merge_idx != -1:
                    merge_table[(idx, idx+1)] = merge_idx

            if len(merge_table) == 0:
                break

            # merge_table 里找个 score 最小的
            idx = min(merge_table.keys(), key=lambda x:merge_table[x])[0]
            bytes_list = bytes_list[:idx] + [bytes_list[idx] + bytes_list[idx+1]] + bytes_list[idx+2:]

        # import pdb
        # pdb.set_trace()

        # print(bytes_list)
        # 合并结束，开始转 token_id

        tokens: list[int] = []
        for b in bytes_list:
            # if b in self.reverse_vocab:
            tokens.append(self.reverse_vocab[b])
            # else:
            #     tokens.append(int(b[0]))

        return tokens

    def encode(self, s: str) -> list[int]:
        # 1. pre_tokenizer
        pretokens = pre_tokenize(s, self.special_tokens)
        print("fuck", pretokens)
        print("special_tokens: ", self.special_tokens)
        print("raw", s)

        # 2. apply the merge(从上到下)
        # 直到没有 merge 可用
        encoded_results = [
            self.encode_pretoken(pretoken)
            for pretoken in pretokens
        ]

        # import pdb
        # pdb.set_trace()

        # 3. 转 int
        result: list[int] = []
        for subresult in encoded_results:
            for token_id in subresult:
                result.append(token_id)

        print(f"s: {s}", f"reult: {result}")

        return result

    def decode(self, tokens: list[int]) -> str:
        # 转 bytes
        bytes_list: list[bytes] = []
        for token in tokens:
            bytes_list.append(self.vocab[token])

        return (b''.join(bytes_list)).decode(encoding="utf-8", errors="replace")

    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        yield from itertools.chain.from_iterable(
            map(self.encode, iterable)
        )
        