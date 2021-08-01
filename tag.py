import numpy as np

from argparse import ArgumentParser
from CLIP import clip

import clipgrams


def main():
    # Args
    parser = ArgumentParser()
    parser.add_argument('--textfile', type=str)
    parser.add_argument('--index_dir', type=str)
    parser.add_argument('--img', type=str, default=None)
    parser.add_argument('--topk', type=int, default=10)
    parser.add_argument('--clip_model', type=str, default='ViT-B/16')
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()

    # Load list
    text = clipgrams.load_text(args)

    # Load index
    index = clipgrams.load_index(args)
    
    # Load clip
    net, preprocess = clip.load(args.clip_model, jit=False)
    net = net.eval().requires_grad_(False).to(args.device)

    # Load image
    img, xq = clipgrams.encode_single_image(args, net, preprocess)
    
    # Search!
    D, I = index.search(xq, args.topk) 

    # Display result
    print("\nTop predictions:\n")
    results = [text[idx] for idx in I[0]]
    maxlen = max([len(x) for x in results])
    for i, r in enumerate(results):
        print(f"{r:>{maxlen}s}: {D[0][i]:.2f}")


if __name__ == '__main__':
    main()