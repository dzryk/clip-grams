import collections
import glob
import nltk
import torch
import numpy as np
import faiss
import requests
import io

from CLIP import clip
from nltk.tokenize import wordpunct_tokenize
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from torch.utils.data import DataLoader


class TextDataset(torch.utils.data.Dataset):
    def __init__(self,
                 folder : str,
                 args):
        super().__init__()
        self.prefix = args.prefix
        self.lower = args.lower
        self.tokenizer = lambda text: clip.tokenize([text], truncate=True)[0]

        path = Path(folder)
        text_files = sorted([*path.glob('**/*.txt')])

        data = []
        tokens = []
        for f in text_files:
            try:
                text = f.read_text().split('\n')
            except UnicodeDecodeError:
                continue
            text = list(filter(lambda t: len(t) > 0, text))
            if args.use_unigrams or args.use_bigrams or args.use_trigrams:
                for line in text:
                    if args.lower:
                        tokens.extend([w.lower() for w in wordpunct_tokenize(line)])
                    else:
                        tokens.extend([w for w in wordpunct_tokenize(line)])
            if args.use_line:
                if args.lower:
                    text = [t.lower() for t in text]
                data.extend(text)
    
        if args.use_unigrams:
            unigrams = collections.Counter(tokens).most_common(args.topk)
            unigrams = [t[0] for t in unigrams]
            data.extend(unigrams)
     
        if args.use_bigrams:
            bigram_measures = nltk.collocations.BigramAssocMeasures()
            finder = nltk.collocations.BigramCollocationFinder.from_words(tokens)
            if args.filter:
                finder.apply_freq_filter(args.filter)
            bigrams = finder.nbest(bigram_measures.pmi, args.topk)
            bigrams = [' '.join(g) for g in bigrams]
            data.extend(bigrams)

        if args.use_trigrams:
            trigram_measures = nltk.collocations.TrigramAssocMeasures()
            finder = nltk.collocations.TrigramCollocationFinder.from_words(tokens)
            if args.filter:
                finder.apply_freq_filter(args.filter)
            trigrams = finder.nbest(trigram_measures.pmi, args.topk)
            trigrams = [' '.join(g) for g in trigrams]
            data.extend(trigrams)
        
        print(f'Number of entries: {len(data)}')
        self.data = data

    def __getitem__(self, idx):
        item = self.prefix + self.data[idx]
        return self.tokenizer(item)
    
    def __len__(self):
        return len(self.data)


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


def fetch(url_or_path):
    if str(url_or_path).startswith('http://') or str(url_or_path).startswith('https://'):
        r = requests.get(url_or_path)
        r.raise_for_status()
        fd = io.BytesIO()
        fd.write(r.content)
        fd.seek(0)
        return fd
    return open(url_or_path, 'rb')


def encode(args, net):
    text_embeddings = []
    dataset = TextDataset(folder=args.text_dir, args=args)
    data = DataLoader(dataset,
                      batch_size=args.batch_size,
                      shuffle=False,
                      num_workers=args.num_prepro_workers,
                      pin_memory=True,
                      prefetch_factor=2)
    print('Encoding with CLIP...')
    for item in tqdm(data):
        text_features = net.encode_text(item.to(args.device))
        text_features /= text_features.norm(dim=-1, keepdim=True)
        text_embeddings.append(text_features.cpu().numpy())
    return np.concatenate(text_embeddings)