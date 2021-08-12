import json
import os
import numpy as np

from argparse import ArgumentParser
from autofaiss.external.quantize import Quantizer
from CLIP import clip

import clipgrams


def main():
    # Args
    parser = ArgumentParser()
    parser.add_argument('--text_dir', type=str, default=None)
    parser.add_argument('--index_dir', type=str)
    parser.add_argument('--clip_model', type=str, default='ViT-B/16')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--chunk_size', type=int, default=500000)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_prepro_workers', type=int, default=8)
    parser.add_argument('--prefix', type=str, default='')
    parser.add_argument('--save_entries', type=bool, default=False)
    parser.add_argument('--lower', type=bool, default=True)
    parser.add_argument('--use_line', type=bool, default=False)
    parser.add_argument('--use_unigrams', type=bool, default=False)
    parser.add_argument('--use_bigrams', type=bool, default=False)
    parser.add_argument('--use_trigrams', type=bool, default=False)
    parser.add_argument('--topk_ngrams', type=int, default=10000)
    parser.add_argument('--filter', type=int, default=3)
    parser.add_argument('--metric_type', type=str, default='ip')
    parser.add_argument('--max_index_memory_usage', type=str, default='32GB')
    parser.add_argument('--current_memory_available', type=str, default='32GB')
    parser.add_argument('--max_index_query_time_ms', type=int, default=10)
    args = parser.parse_args()

    # Load clip, compute embeddings and save
    if args.text_dir:
        net, preprocess = clip.load(args.clip_model, jit=False)
        net = net.eval().requires_grad_(False).to(args.device)
        clipgrams.encode(args, net)

        # Store args
        fname = os.path.join(args.index_dir, 'args.txt')
        with open(fname, 'w') as f:
            json.dump(args.__dict__, f, indent=2)

    # Compute index
    quantizer = Quantizer()
    quantizer.quantize(embeddings_path=args.index_dir,
                       output_path=args.index_dir,
                       metric_type=args.metric_type,
                       max_index_memory_usage=args.max_index_memory_usage,
                       current_memory_available=args.current_memory_available,
                       max_index_query_time_ms=args.max_index_query_time_ms)


if __name__ == '__main__':
    main()