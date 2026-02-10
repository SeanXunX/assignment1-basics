import json
import yaml
import argparse

from cs336_basics.bpe_tokenizer import BPETokenizer


def main():
    parser = argparse.ArgumentParser(description="BPE tokenizer trainer.")
    parser.add_argument("config_file", type=str, help="yaml config file path")
    args = parser.parse_args()

    with open(args.config_file, "r") as f:
        config = yaml.safe_load(f)

    tokenizer = BPETokenizer(num_processes=f.num_processes)
    tokenizer.train(input_path=f.dataset, vocab_size=f.vocab_size, special_tokens=f.special_tokens)
    vocab = {i: bs for i, bs in enumerate(tokenizer.vocab)}

    # save result


if __name__ == "__main__":
    main()
