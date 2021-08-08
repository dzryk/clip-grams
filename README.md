# clip-grams

clip-grams is a tool for creating [Faiss](https://github.com/facebookresearch/faiss) knn indices from CLIP embeddings of large text files. It is primarily designed for:

- Image tagging
- Analyzing CLIP's ability to describe images with different categories of text

We make use of [autofaiss](https://github.com/criteo/autofaiss) to automatically estimate the search parameters.

## Quickstart

A [colab](https://colab.research.google.com/drive/19e7kbE9s4voya6s668vl5eafpIZh-5BC?usp=sharing) is available that illustrates the key functions on a small collection of images and text.

## Getting started

Install requirements:

```
pip install -r requirements.txt
```

Clone CLIP into this project's repository:

```
git clone https://github.com/openai/CLIP
```

There are two main functions. Suppose we have a folder with text files from which an index is to be constructed. To compute a Faiss index using autofaiss:

```
python3 index.py --text_dir=[path to text folder] --index_dir=[folder to store index] --use_line=true
```

The `--use_line` argument indicates that each line from each text file is considered an entry. We can also pass arguments to include unigrams, bigrams and trigrams. The `--topk_ngrams` argument sets an upper bound on the number of n-gram entries. The `--filter` argument will only include n-grams that occur at least that many times in the corpus. A prefix can be passed for CLIP encoding using the `--prefix` argument. The `--chunk_size` argument specifies approximately how many entries each npy file of CLIP embeddings will have. Several arguments are also directly passed to autofaiss for index construction. See `index.py` for full list of arguments.

If you've already created an index and want to create another without having to re-compute npy files, don't pass anything to `--text_dir`. It will immediately skip to computing a new index.

Once an index is created, we can use it to tag all images that occur in a directory:

```
python3 tag.py --image_dir=[path to image folder] --index_dir=[folder where index lives] --knn=5
```

This will create a new file with a `.knn` extension for each image in the directory, using the same stem. The `--knn` argument will return the top-k ranked entries from the index. See `tag.py` for full list of arguments.

## Acknowledgements

Thanks to Christoph S. from EleutherAI for data processing.

The [clip-retrieval](https://github.com/rom1504/clip-retrieval) repo from rom1504, from which a lot of this code has been inspired from.

## TODO
- [x] Handle more general input types
- [x] Batch/Dataset tagging
- [ ] Multi-GPU inference (low priority for now)
