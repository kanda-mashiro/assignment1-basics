import os
import regex
import time
import tqdm

from collections import defaultdict
from multiprocessing import Pool

from cs336_basics import PAT
from cs336_basics.pretokenization_example import find_chunk_boundaries
from cs336_basics.timer_utils import timer
from tests.common import gpt2_bytes_to_unicode


def pretokenization(input_path, start, end, special_tokens) -> dict[str, int]:
    chunks: dict[str, int] = defaultdict(int)
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode('utf-8')
        pattern = "|".join([regex.escape(token) for token in special_tokens])
        for item in regex.split(pattern, chunk):
            if len(item) != 0:
                for x in regex.findall(PAT, item):
                    chunks[x] += 1
    return chunks

@timer
def run_train_bpe_v2(
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
    start = time.time()
    desired_num_chunks = 1000
    with open(input_path, "rb") as f:
        # Q: 只考虑 <|endoftext|> 是否可行？是否需要考虑所有的 special_tokens？
        boundaries = find_chunk_boundaries(f, desired_num_chunks, b"<|endoftext|>")

    pool = Pool(10)
    results = pool.starmap(pretokenization, [
        (input_path, start, end, special_tokens) for (start, end) in zip(boundaries[:-1], boundaries[1:])])
    chunks: dict[str, int] = defaultdict(int)
    for result in results:
        for chunk, k in result.items():
            chunks[chunk] += k
    end = time.time()
    print(f"pretokenzation: {end-start:.4f} s")

    vocab: dict[int, bytes] = {}
    merges: list[tuple[bytes, bytes]] = []

    # 初始化词表
    # 1. special_token
    # 2. 0-255
    for special_token in special_tokens:
        vocab[len(vocab)] = special_token.encode('utf-8')
    for idx in gpt2_bytes_to_unicode().keys():
        vocab[len(vocab)] = bytes([idx])
    assert len(vocab) == len(special_tokens) + 256

    # 维护 str 到 list(bytes) 的映射
    # 当发生合并时，不更新 key 只更新 value
    hash: dict[str, list(bytes)] = {}
    for chunk, _ in chunks.items():
        hash[chunk] = [bytes([b]) for b in chunk.encode("utf-8")]

    # 统计每个 pair 对出现的频率，并寻找频率最高的一个
    counts: dict[tuple[bytes, bytes], int] = defaultdict(int)
    # 预先计算一轮，将 counts 的内容填充
    # 后续通过增量更新来降低算法复杂度，从而提升性能
    for chunk, k in chunks.items():
        for pair in zip(hash[chunk][:-1], hash[chunk][1:]):
            counts[pair] += k

    for _ in tqdm.tqdm(range(0, vocab_size - len(vocab))):
        # 先按照频率排序，如果频率相同，则按照字典序排序
        max_freq = max(counts.items(), key=lambda x:(x[1], x[0]))

        # 每一轮循环都会产生一个新的 token
        token1, token2 = max_freq[0]
        new_token = token1 + token2
        vocab[len(vocab)] = new_token

        # 添加到 merge 记录中
        merges.append((token1, token2))

        # 移除已经被选出来的 pair 对
        del counts[(token1, token2)]

        def merge_helper(t1, t2):
            for chunk, k in chunks.items():
                bs = hash[chunk]

                # 先扫描一遍，判断是否需要更新 bs
                need_update = False
                for pair in zip(bs[:-1], bs[1:]):
                    if pair[0] == t1 and pair[1] == t2:
                        need_update = True
                        break

                if not need_update:
                    continue

                idx = 0
                new_bs: list[bytes] = []
                while idx < len(bs):
                    if idx + 1 < len(bs) and bs[idx] == t1 and bs[idx+1] == t2:
                        new_bs.append(t1 + t2)
                        if idx - 1 >= 0:
                            counts[(bs[idx-1], t1)] -= k
                        if idx + 2 < len(bs):
                            counts[(t2, bs[idx+2])] -= k

                        idx += 2
                        need_update = True
                    else:
                        new_bs.append(bs[idx])
                        idx += 1

                for pair in zip(new_bs[:-1], new_bs[1:]):
                    new_token = t1 + t2
                    if pair[0] == new_token or pair[1] == new_token:
                        counts[pair] += k
                hash[chunk] = new_bs

        merge_helper(token1, token2)

    return (vocab, merges)