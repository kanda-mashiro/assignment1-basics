from adapters import run_train_bpe_v1, run_train_bpe_v2
from common import gpt2_bytes_to_unicode

# v1, m1 = run_train_bpe_v1("./tests/fixtures/tinystories_sample_5M.txt", 300, ["<|endoftext|>"])
# v2, m2 = run_train_bpe_v2("./tests/fixtures/tinystories_sample_5M.txt", 300, ["<|endoftext|>"])
v1, m1 = run_train_bpe_v1("./tests/fixtures/corpus.en", 300, ["<|endoftext|>"])
# v2, m2 = run_train_bpe_v2("./tests/fixtures/corpus.en", 500, ["<|endoftext|>"])
print(v1)
for k, v in v1.items():
    print(k, v)

# assert v1 == v2
# assert v2 == v3
# print(v2)
# print(v3)

