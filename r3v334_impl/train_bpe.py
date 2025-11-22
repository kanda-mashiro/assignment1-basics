import time
import regex as rex
import re
from multiprocessing import Pool
import os

from cs336_basics.pretokenization_example import find_chunk_boundaries

chunk_cnt = {}
pair_cnt = {}

def chunk2pairCount(chunk: list[bytes]) -> dict[tuple[bytes, bytes], int]:
        all_pair = zip(chunk[0:-1], chunk[1:])
        result: dict[tuple[bytes, bytes], int] = {}
        s = list_bytes_to_str(chunk)
        s_cnt = chunk_cnt[s]

        for pair in all_pair:
            if pair not in result:
                result[pair] = 0
            result[pair] += s_cnt

        return result

def merge_pairCount(
    left: dict[tuple[bytes, bytes], int], right: dict[tuple[bytes, bytes], int]
) -> dict[tuple[bytes, bytes], int]:
    result = left
    for pair in right:
        cnt = right[pair]
        if pair in result:
            result[pair] += cnt
        else:
            result[pair] = cnt
    return result

def list_bytes_to_str(b: list[bytes]) -> str:
    return f'{b}'

def bytes_list_to_pair(bytes_list: list[bytes]) -> list[tuple[bytes, bytes]]:
    return list(zip(bytes_list[0:-1], bytes_list[1:]))

def find_freq_pair() -> tuple[bytes, bytes]:
        max_cnt = max(pair_cnt.values())
        max_pairs = []
        for pair, cnt in pair_cnt.items():
            if cnt != max_cnt:
                continue
            max_pairs.append(pair)
        return max(max_pairs)


# 替换 byte_chunks 里对应的 内容
def replace_token(target_pair: tuple[bytes, bytes]):
    """
    key: str
    value: {
        "cnt": {chunk_cnt}
        "bytes_list": ...
    }
    """
    new_token = target_pair[0] + target_pair[1]
    for chunk in chunk_cnt:
        cnt = chunk_cnt[chunk]["cnt"]
        bytes_list = chunk_cnt[chunk]["bytes_list"]
        new_bytes_list = []
        idx = 0
        replace_flag = False
        while idx < len(bytes_list):
            if idx + 1 < len(bytes_list) and \
                    bytes_list[idx] == target_pair[0] and bytes_list[idx+1] == target_pair[1]:

                new_bytes_list.append(new_token)
                idx += 2
                replace_flag = True
                continue

            new_bytes_list.append(bytes_list[idx])
            idx += 1
        
        if replace_flag:
            for pair in bytes_list_to_pair(bytes_list):
                pair_cnt[pair] -= cnt
            
            for pair in bytes_list_to_pair(new_bytes_list):
                if pair not in pair_cnt:
                    pair_cnt[pair] = 0
                pair_cnt[pair] += cnt

        # print(chunk, '|', new_token, '|', new_bytes_list)
        chunk_cnt[chunk]["bytes_list"] = new_bytes_list


def pre_tokenize(input_path: str, start: int, end: int, special_tokens: list[str]) -> dict[str, dict]:
    item_cnt = {}

    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8")
        pattern = "|".join([re.escape(token) for token in special_tokens])
        for item in re.split(pattern, chunk):
            if len(item) == 0:
                continue
            PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
            for chunk in rex.findall(PAT, item):
                if chunk not in item_cnt:
                    byte_chunk = chunk.encode("utf-8")
                    bytes_list = [byte_chunk[idx : idx + 1] for idx in range(len(byte_chunk))]
                    item_cnt[chunk] = {
                        "bytes_list": bytes_list,
                        "cnt": 0
                    }
                item_cnt[chunk]["cnt"] += 1

    return item_cnt

def multi_run_wrapper(args):
    yappi.set_clock_type("cpu")
    yappi.start()
    result = pre_tokenize(*args)
    yappi.stop()
    yappi.get_func_stats().save(f'yappi_{os.getpid()}.prof', type='pstat')
    return result

def train_bpe(input_path: str, special_tokens: list[str], vocab_size: int):
    # 从文件中读取字符串
    # 1. 按照 special_token 去寻找边界，避免将 special_token 拆分成多个部分
    # 2. 移除 special_token

    desired_num_chunks = 1000
    with open(input_path, "rb") as f:
        # Q: 只考虑 <|endoftext|> 是否可行？是否需要考虑所有的 special_tokens？
        boundaries = find_chunk_boundaries(f, desired_num_chunks, b"<|endoftext|>")
    

    global chunk_cnt, pair_cnt
    chunk_cnt = {}
    pair_cnt = {}
    start = time.time()
    with Pool(processes=10) as p:
        chunk_cnts = p.map(
            multi_run_wrapper, 
            [
                (input_path, bound[0], bound[1], special_tokens)
                for bound in list(zip(boundaries[:-1], boundaries[1:]))
            ]
        )

        for item_cnt in chunk_cnts:
            for chunk in item_cnt:
                if chunk not in chunk_cnt:
                    chunk_cnt[chunk] = item_cnt[chunk]
                else:
                    chunk_cnt[chunk]["cnt"] += item_cnt[chunk]["cnt"]

        # chunk_cnt = multi_run_wrapper((input_path, boundaries[0], boundaries[1], special_tokens))    
    print(time.time() - start)
    vocab: dict[int, bytes] = {}
    merges: list[tuple[bytes, bytes]] = []
    # return (vocab, merges)


    _id = -1
    def get_new_id() -> int:
        nonlocal _id
        _id += 1
        return _id

    # 1. 初始化词表
    tokens = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))

    # 1.2.2 control token
    for b in range(2**8):
        if b not in tokens:
            tokens.append(b)

    vocab.update(
        {
            get_new_id(): special_token.encode("utf-8")
            for special_token in special_tokens
        }
    )
    vocab.update(
        {
            get_new_id(): bytes([token])
            for token in tokens
        }
    )

    for chunk in chunk_cnt:
        cnt = chunk_cnt[chunk]["cnt"]
        
        for pair in bytes_list_to_pair(chunk_cnt[chunk]["bytes_list"]):
            if pair not in pair_cnt:
                pair_cnt[pair] = 0
            pair_cnt[pair] += cnt

    # print(pair_cnt)

    while len(vocab) < vocab_size:
        print(len(vocab), "target: ", vocab_size)
        max_pair = find_freq_pair()
        new_token = max_pair[0] + max_pair[1]
        vocab[get_new_id()] = new_token
        merges.append(max_pair)
        replace_token(max_pair)

    return (vocab, merges)
