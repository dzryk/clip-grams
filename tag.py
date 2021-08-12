import json
import os
import numpy as np

from argparse import ArgumentParser
from CLIP import clip

import clipgrams


def main():
    # Args
    parser = ArgumentParser()
    parser.add_argument('--image_dir', type=str)
    parser.add_argument('--index_dir', type=str)
    parser.add_argument('--knn', type=int, default=10)
    parser.add_argument('--clip_model', type=str, default='ViT-B/16')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_prepro_workers', type=int, default=8)
    parser.add_argument('--lower', type=bool, default=True)
    parser.add_argument('--load_entries', type=bool, default=False)
    args = parser.parse_args()

    # Load index args and add to current args
    fname = os.path.join(args.index_dir, 'args.txt')
    with open(fname, 'r') as f:
        index_args = json.load(f)
        for key in list(index_args.keys()):
            if key not in args.__dict__.keys():
                args.__dict__[key] = index_args[key]

    # Load clip
    net, preprocess = clip.load(args.clip_model, jit=False)
    net = net.eval().requires_grad_(False).to(args.device)

    # Tagger
    clipgrams.tagger(args, net)


if __name__ == '__main__':
    main()