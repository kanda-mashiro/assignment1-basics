from adapters import run_train_bpe


if __name__ == '__main__':
    # vocab, merges = run_train_bpe("./tests/fixtures/tinystories_sample_5M.txt", 300, ["<|endoftext|>"])
    vocab, merges = run_train_bpe("/Users/r3v334/code/assignment1-basics/TinyStoriesV2-GPT4-train.txt", 10000, ["<|endoftext|>"])