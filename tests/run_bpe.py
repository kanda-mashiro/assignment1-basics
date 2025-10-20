from adapters import run_train_bpe_v1, run_train_bpe_v2
from common import gpt2_bytes_to_unicode

if __name__ == '__main__':
    # v1, m1 = run_train_bpe_v1("./tests/fixtures/tinystories_sample_5M.txt", 300, ["<|endoftext|>"])
    # v2, m2 = run_train_bpe_v2("./tests/fixtures/tinystories_sample_5M.txt", 300, ["<|endoftext|>"])
    # v1, m1 = run_train_bpe_v3("./tests/fixtures/corpus.en", 500, ["<|endoftext|>"])
    v1, m1 = run_train_bpe_v2("./data/TinyStoriesV2-GPT4-train.txt", 260, ["<|endoftext|>"])
    # v2, m2 = run_train_bpe_v2("./tests/fixtures/corpus.en", 500, ["<|endoftext|>"])