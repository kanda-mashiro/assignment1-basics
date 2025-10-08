from adapters import run_train_bpe

vocab, merges = run_train_bpe("./tests/fixtures/tinystories_sample_5M.txt", 300, ["<|endoftext|>"])