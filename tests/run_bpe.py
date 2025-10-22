from adapters import run_train_bpe

if __name__ == '__main__':
    v, m = run_train_bpe("./tests/fixtures/corpus.en", 500, ["<|endoftext|>"])
    # v, m = run_train_bpe("./tests/fixtures/tinystories_sample_5M.txt", 300, ["<|endoftext|>"])
    # v, m = run_train_bpe("./data/TinyStoriesV2-GPT4-train.txt", 10000, ["<|endoftext|>"])
    # v, m = run_train_bpe("./data/owt_train.txt", 32000, ["<|endoftext|>"])