import json
import os
import numpy as np

from argparse import ArgumentParser
from CLIP import clip

import clipgrams


def main():
    # Args
    parser = ArgumentParser()
    parser.add_argument('--index_dir', type=str)
    parser.add_argument('--img', type=str, default=None)
    parser.add_argument('--knn', type=int, default=10)
    parser.add_argument('--clip_model', type=str, default='ViT-B/16')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--lower', type=bool, default=True)
    args = parser.parse_args()

    # Load index args and add to current args
    fname = os.path.join(args.index_dir, 'args.txt')
    with open(fname, 'r') as f:
        index_args = json.load(f)
        for key in list(index_args.keys()):
            if key not in args.__dict__.keys():
                args.__dict__[key] = index_args[key]

    # Load list
    text = clipgrams.TextDataset(folder=args.text_dir, args=args).data

    # Load index
    index = clipgrams.load_index(args)
    
    # Load clip
    net, preprocess = clip.load(args.clip_model, jit=False)
    net = net.eval().requires_grad_(False).to(args.device)

    # Load image
    img, xq = clipgrams.encode_single_image(args, net, preprocess)
    
    # Search!
    D, I = index.search(xq, args.knn) 

    # Display result
    print("\nTop predictions:\n")
    results = [text[idx] for idx in I[0]]
    maxlen = max([len(x) for x in results])
    for i, r in enumerate(results):
        print(f"{r:>{maxlen}s}: {D[0][i]:.2f}")


if __name__ == '__main__':
    main()