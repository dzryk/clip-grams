import os
import numpy as np

from argparse import ArgumentParser
from autofaiss.external.quantize import Quantizer
from CLIP import clip

import clipgrams


def main():
    # Args
    parser = ArgumentParser()
    parser.add_argument('--textfile', type=str)
    parser.add_argument('--index_dir', type=str)
    parser.add_argument('--clip_model', type=str, default='ViT-B/16')
    parser.add_argument('--prefix', type=str, default='')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_prepro_workers', type=int, default=8)
    parser.add_argument('--lower', type=bool, default=False)
    parser.add_argument('--metric_type', type=str, default='ip')
    parser.add_argument('--max_index_memory_usage', type=str, default='32GB')
    parser.add_argument('--current_memory_available', type=str, default='32GB')
    parser.add_argument('--max_index_query_time_ms', type=int, default=10)
    args = parser.parse_args()

    # Load list
    text = clipgrams.load_text(args)

    # Load clip
    net, preprocess = clip.load(args.clip_model, jit=False)
    net = net.eval().requires_grad_(False).to(args.device)

    # Compute embeddings and save
    print(f'Number of entries: {len(text)}')
    print('Encoding with CLIP...')
    emb = clipgrams.encode(args, net, text)
    fname = os.path.join(args.index_dir, 'emb.npy')
    np.save(fname, emb)

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