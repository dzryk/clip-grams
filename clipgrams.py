import pickle
import glob
import torch
import numpy as np
import faiss
import requests
import io

from CLIP import clip
from tqdm import tqdm
from PIL import Image
from torch.utils.data import DataLoader


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, args, data):
        self.data = data
        self.prefix = args.prefix
        self.lower = args.lower
        self.tokenizer = lambda text: clip.tokenize([text], truncate=True)[0]

    def __getitem__(self, idx):
        item = self.prefix + self.data[idx]
        if self.lower:
            item = item.lower()
        return self.tokenizer(item)
    
    def __len__(self):
        return len(self.data)


def fetch(url_or_path):
    if str(url_or_path).startswith('http://') or str(url_or_path).startswith('https://'):
        r = requests.get(url_or_path)
        r.raise_for_status()
        fd = io.BytesIO()
        fd.write(r.content)
        fd.seek(0)
        return fd
    return open(url_or_path, 'rb')


def load_text(args):
    with open(args.textfile, 'rb') as f:
        text = pickle.load(f)
    return text


def load_index(args):
    index = faiss.read_index(glob.glob(f"{args.index_dir}/*.index")[0])
    return index


def encode_single_image(args, net, preprocess):
    img = Image.open(fetch(args.img))
    mat = preprocess(img).unsqueeze(0).to(args.device)
    xq = net.encode_image(mat)
    xq /= xq.norm(dim=-1, keepdim=True)
    xq = xq.cpu().numpy().astype('float32')
    return img, xq


def encode(args, net, text):
    text_embeddings = []
    dataset = TextDataset(args, text)
    data = DataLoader(dataset,
                      batch_size=args.batch_size,
                      shuffle=False,
                      num_workers=args.num_prepro_workers,
                      pin_memory=True,
                      prefetch_factor=2)
    for item in tqdm(data):
        text_features = net.encode_text(item.to(args.device))
        text_features /= text_features.norm(dim=-1, keepdim=True)
        text_embeddings.append(text_features.cpu().numpy())
    return np.concatenate(text_embeddings)