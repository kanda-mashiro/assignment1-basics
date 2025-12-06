import json
from multiprocessing import Pool
import time
from datasets import load_dataset
import torch
import numpy as np
from r3v334_impl import dataloader
from r3v334_impl.adamw import RAdamW
from r3v334_impl.modules import RTransformerLM
from r3v334_impl.utils import r_annealing_lr, r_cross_entropy, r_gradient_clipping, r_load_checkpoint, r_save_checkpoint
from tests.test_tokenizer import MERGES_PATH, VOCAB_PATH, get_tokenizer_from_vocab_merges_path

# CKPT_PATH = "./ckpts/llm_init.ckpt"
CKPT_PATH = ""
MAX_ITERATION = 1000

if __name__ == "__main__":
    tok = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH,
        merges_path=MERGES_PATH,
        special_tokens=["<|endoftext|>"]
    )

    llm = RTransformerLM(
        vocab_size=len(tok.vocab),
        context_length=16,
        d_model=64,
        num_layers=4,
        num_heads=4,
        d_ff=128,
        rope_theta=10000.0,
        device="mps",
    )

    opt = RAdamW(
        params=llm.parameters(True),
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
    )

    iteration_count = 0


    # load checkpoint
    if not CKPT_PATH:
        # create a init ckpt
        r_save_checkpoint(llm, opt, iteration_count, "./ckpts/llm_init.ckpt")
        print("save init checkpont")
    else:
        iteration_count = r_load_checkpoint(CKPT_PATH, llm, opt)

    print(f"current iteration: {iteration_count}")


    # read_data
    tokens = json.load(open("./token_ids", "r"))

    # MAX_ITERATION epoch
    for i in range(iteration_count, MAX_ITERATION):
        start = time.time()
        
        lr = 1e-2
        batch_size = 256
        for _ in range(len(tokens) // (batch_size * 16)):
            train_data, label = dataloader.r_load_data(
                    np.asarray(tokens, dtype=np.int32), batch_size, 16, "cpu")
        
            opt.zero_grad()
            logits = llm.forward(train_data)

            loss = r_cross_entropy(logits.view(-1, logits.shape[-1]), label.view(-1))
            print(loss.cpu().item())
            loss.backward()
            r_gradient_clipping(llm.parameters(True), 1e-2)

            # lr = opt.param_groups[0]["alpha"]
            opt.param_groups[0]["alpha"] = lr
            opt.step()

        total_cost = time.time() - start

        print(f"it={i}, "\
                f"cost={total_cost}s, loss={loss.cpu().item()}, lr={lr}")
        r_save_checkpoint(llm, opt, i, f"./ckpts/llm_{i}.ckpt")
        print(f"save checkpoint {i}")
