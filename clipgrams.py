import collections
import os
import glob
import nltk
import torch
import numpy as np
import faiss
import requests
import io
import PIL

from CLIP import clip
from nltk.tokenize import wordpunct_tokenize
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms as T


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
            unigrams = collections.Counter(tokens).most_common(args.topk_ngrams)
            unigrams = [t[0] for t in unigrams]
            data.extend(unigrams)
     
        if args.use_bigrams:
            bigram_measures = nltk.collocations.BigramAssocMeasures()
            finder = nltk.collocations.BigramCollocationFinder.from_words(tokens)
            if args.filter:
                finder.apply_freq_filter(args.filter)
            bigrams = finder.nbest(bigram_measures.pmi, args.topk_ngrams)
            bigrams = [' '.join(g) for g in bigrams]
            data.extend(bigrams)

        if args.use_trigrams:
            trigram_measures = nltk.collocations.TrigramAssocMeasures()
            finder = nltk.collocations.TrigramCollocationFinder.from_words(tokens)
            if args.filter:
                finder.apply_freq_filter(args.filter)
            trigrams = finder.nbest(trigram_measures.pmi, args.topk_ngrams)
            trigrams = [' '.join(g) for g in trigrams]
            data.extend(trigrams)
        
        print(f'Number of entries: {len(data)}')
        self.data = data

    def __getitem__(self, idx):
        item = self.prefix + self.data[idx]
        return self.tokenizer(item), self.data[idx]
    
    def __len__(self):
        return len(self.data)


def dl_collate_fn(batch):
    return torch.stack([row[0] for row in batch]), [row[1] for row in batch]


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self,
                 folder: str,
                 image_size=224):
        super().__init__()
        path = Path(folder)

        image_files = sorted([
            *path.glob('**/*.png'), *path.glob('**/*.jpg'),
            *path.glob('**/*.jpeg'), *path.glob('**/*.bmp')
        ])

        self.image_files = {image_file.stem: image_file for image_file in image_files}
        self.keys = list(self.image_files.keys())
        self.image_transform = T.Compose([
            T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(image_size),
            T.Lambda(self.fix_img),
            T.ToTensor(),
            T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])

    def __len__(self):
        return len(self.keys)

    def fix_img(self, img):
        return img.convert('RGB') if img.mode != 'RGB' else img

    def sequential_sample(self, ind):
        if ind >= self.__len__() - 1:
            return self.__getitem__(0)
        return self.__getitem__(ind + 1)

    def skip_sample(self, ind):
        if self.shuffle:
            return self.random_sample()
        return self.sequential_sample(ind=ind)

    def __getitem__(self, ind):
        key = self.keys[ind]
        image_file = self.image_files[key]
        knn_file = image_file.with_suffix('.knn')

        try:
            image_tensor = self.image_transform(PIL.Image.open(image_file))
        except (PIL.UnidentifiedImageError, OSError) as corrupt_image_exceptions:
            print(f"An exception occurred trying to load file {image_file}.")
            print(f"Skipping index {ind}")
            return self.skip_sample(ind)

        return image_tensor, knn_file


def load_index(args):
    index = faiss.read_index(glob.glob(f"{args.index_dir}/*.index")[0])
    return index


def encode(args, net):
    text_embeddings = []
    entries = []
    dataset = TextDataset(folder=args.text_dir, args=args)
    data = DataLoader(dataset,
                      batch_size=args.batch_size,
                      shuffle=False,
                      num_workers=args.num_prepro_workers,
                      pin_memory=True,
                      prefetch_factor=2)
    print('Encoding with CLIP...')
    batches_seen = 0
    chunks_count = 0
    batches_per_chunk = args.chunk_size // args.batch_size
    for item, entry in tqdm(data):
        text_features = net.encode_text(item.to(args.device))
        text_features /= text_features.norm(dim=-1, keepdim=True)
        text_embeddings.append(text_features.cpu().numpy().astype('float32'))
        batches_seen += 1
        if batches_seen % batches_per_chunk == 0:
            emb = np.concatenate(text_embeddings)
            idx = str(chunks_count).zfill(5)
            fname = os.path.join(args.index_dir, f'emb{idx}.npy')
            np.save(fname, emb)
            chunks_count += 1
            text_embeddings = []
        if args.save_entries:
            entries.extend(entry)
    emb = np.concatenate(text_embeddings)
    idx = str(chunks_count).zfill(5)
    fname = os.path.join(args.index_dir, f'emb{idx}.npy')
    np.save(fname, emb)

    # Store entries if applicable
    if args.save_entries:
        print('Saving entries...')
        fname = os.path.join(args.index_dir, 'entries.txt')
        with open(fname, 'w') as f:
            for line in entries:
                f.write(f'{line}\n')


def tagger(args, net):
    if args.load_entries:
        text = []
        fname = os.path.join(args.index_dir, 'entries.txt')
        with open(fname, 'r') as f:
            text.extend([line.strip() for line in f])
    else:
        text = TextDataset(folder=args.text_dir, args=args).data
    index = faiss.read_index(glob.glob(f"{args.index_dir}/*.index")[0])
    dataset = ImageDataset(folder=args.image_dir)
    data = DataLoader(dataset,
                      batch_size=args.batch_size,
                      shuffle=False,
                      num_workers=args.num_prepro_workers,
                      pin_memory=True,
                      collate_fn=dl_collate_fn,
                      prefetch_factor=2)
    print('Tagging images...')
    for imgs, paths in tqdm(data):
        xq = net.encode_image(imgs.to(args.device))
        xq /= xq.norm(dim=-1, keepdim=True)
        xq = xq.cpu().numpy().astype('float32')
        indices = index.search(xq, args.knn)[1]
        for idx in range(len(xq)):
            result = ''.join(f'{text[i]}\n' for i in indices[idx])
            paths[idx].write_text(result)