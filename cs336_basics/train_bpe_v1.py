import os
import regex
import tqdm

from collections import defaultdict

from cs336_basics import PAT
from cs336_basics.pretokenization_example import find_chunk_boundaries
from cs336_basics.timer_utils import timer
from tests.common import gpt2_bytes_to_unicode

@timer
def run_train_bpe_v1(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    # 从文件中读取字符串
    # 1. 按照 special_token 去寻找边界，避免将 special_token 拆分成多个部分
    # 2. 移除 special_token
    chunks: list[bytes] = []
    desired_num_chunks = 4
    with open(input_path, "rb") as f:
        # Q: 只考虑 <|endoftext|> 是否可行？是否需要考虑所有的 special_tokens？
        boundaries = find_chunk_boundaries(f, desired_num_chunks, b"<|endoftext|>")
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode('utf-8')
            pattern = "|".join([regex.escape(token) for token in special_tokens])
            for item in regex.split(pattern, chunk):
                item = item.strip()
                if len(item) != 0:
                    for x in regex.findall(PAT, item):
                        chunks.append(x.encode('utf-8'))


    vocab: dict[int, bytes] = {}
    merges: list[tuple[bytes, bytes]] = []

    # 初始化词表
    # 1. special_token
    # 2. 0-255
    for special_token in special_tokens:
        vocab[len(vocab)] = special_token.encode('utf-8')
    gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
    for _, token in gpt2_bytes_to_unicode().items():
        vocab[len(vocab)] = bytes([gpt2_byte_decoder[t] for t in token])
    assert len(vocab) == len(special_tokens) + 256

    # 转换 chunks 的格式，便于操作
    # list[bytes] -> list[list[bytes]]
    new_chunks: list[list[bytes]] = []
    for chunk in chunks:
        new_chunks.append([chunk[idx:idx+1] for idx in range(len(chunk))])

    for _ in tqdm.tqdm(range(0, vocab_size - len(vocab))):
        # 统计每个 pair 对出现的频率，并寻找频率最高的一个
        counts: dict[tuple[bytes, bytes], int] = defaultdict(int)
        for chunk in new_chunks:
            for pair in zip(chunk[:-1], chunk[1:]):
                counts[pair] += 1
        # 先按照频率排序，如果频率相同，则按照字典序排序
        max_freq = max(counts.items(), key=lambda x:(x[1], x[0]))

        # 每一轮循环都会产生一个新的 token
        token1, token2 = max_freq[0]
        new_token = token1 + token2
        vocab[len(vocab)] = new_token
        merges.append((token1, token2))

        def merge_helper(chunks, t1, t2):
            new_chunks: list[list[bytes]] = []
            for chunk in chunks:
                idx = 0
                new_chunk: list[bytes] = []
                while idx < len(chunk):
                    if idx + 1 < len(chunk) and chunk[idx] == t1 and chunk[idx+1] == t2:
                        new_chunk.append(t1 + t2)
                        idx += 2
                    else:
                        new_chunk.append(chunk[idx])
                        idx += 1
                new_chunks.append(new_chunk)
            return new_chunks
        # 合并 new_chunks 中的相关 token
        new_chunks = merge_helper(new_chunks, token1, token2)

    return (vocab, merges)