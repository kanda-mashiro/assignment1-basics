from adapters import run_train_bpe_v1, run_train_bpe_v2

_, _ = run_train_bpe_v1("./tests/fixtures/tinystories_sample_5M.txt", 300, ["<|endoftext|>"])
_, _ = run_train_bpe_v2("./tests/fixtures/tinystories_sample_5M.txt", 300, ["<|endoftext|>"])