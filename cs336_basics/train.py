import argparse
import tqdm
import numpy as np
import torch
import os

from cs336_basics.components import AdamW, TransformerLM, cross_entropy, gen_dataset, save_checkpoint, softmax
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.train_bpe_v2 import run_train_bpe_v2 as run_train_bpe


def main(args):
    device = args.device

    # tokenizer
    special_tokens: list[str] = []
    if args.special_tokens_path:
        with open(args.special_tokens_path, 'r', encoding='utf-8') as f:
            for token in f.readlines():
                token = token.strip()
                if len(token) == 0:
                    continue
                special_tokens.append(token)
    else:
        special_tokens.append('<|endoftext|>')

    vocab, merges = run_train_bpe(args.dataset_path, args.vocab_size, special_tokens)
    tokenizer = Tokenizer(vocab, merges, special_tokens)

    # dataloader
    with open(args.dataset_path, 'r', encoding='utf-8') as f:
        tokens = [token for token in tokenizer.encode_iterable(f)]
    print(f'tokens: {len(tokens)}')

    # model
    lm = TransformerLM(
        args.vocab_size,
        args.context_length,
        args.d_model,
        args.num_layers,
        args.num_heads,
        args.d_ff,
        args.rope_theta,
        device)

    # opt
    opt = AdamW(lm.parameters(), args.learning_rate, args.weight_decay, [args.min_beta, args.max_beta], args.eps)

    # train
    bs = args.batch_size
    for epoch in range(args.epoch):
        count = len(tokens)//(bs*args.context_length)
        for it in tqdm.tqdm(range(count)):
            lm.train()
            data, label = gen_dataset(np.asarray(tokens, dtype=np.int32), bs, args.context_length, device)
            output = lm.forward(data)
            opt.zero_grad()

            loss = cross_entropy(output.view(-1, output.shape[-1]), label.view(-1))
            loss.backward()
            opt.step()

        lm.eval()
        text = "I eat"
        text_tokens = tokenizer.encode(text)
        while len(text_tokens) < 10:
            output = lm.forward(text_tokens)[-1]
            top1 = torch.argmax(softmax(output, -1)).item()
            text_tokens.append(top1)
        seq = tokenizer.decode(text_tokens)
        print(f'loss: {loss}, seq: {seq}')

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    save_checkpoint(lm, opt, 0, args.save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='cs336 train')
    parser.add_argument('--dataset_path', type=str)
    parser.add_argument('--vocab_size', type=int, default=1000)
    parser.add_argument('--special_tokens_path', type=str)
    parser.add_argument('--context_length', type=int, default=256)

    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=8)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--d_ff', type=int, default=512)
    parser.add_argument('--rope_theta', type=float, default=10000.0)

    parser.add_argument('--learning_rate', '-lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', '-wd', type=float, default=0.01)
    parser.add_argument('--min_beta', type=float, default=0.90)
    parser.add_argument('--max_beta', type=float, default=0.95)
    parser.add_argument('--eps', type=float, default=1e-8)

    parser.add_argument('--batch_size', '-bs', type=int, default=32)
    parser.add_argument('--epoch', type=int, default=10000)
    parser.add_argument('--device', '-d', type=str, default='cpu')

    parser.add_argument('--save_path', type=str, default='./checkpoints/lm.cp')
    main(parser.parse_args())

