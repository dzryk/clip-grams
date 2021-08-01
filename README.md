# clip-grams

clip-grams is a tool for creating Faiss knn indices from CLIP embeddings of large text lists. It is primarily designed for:

- Image tagging
- Analyzing CLIP's ability to describe images with different categories of text

(WIP)

## Getting started

Install requirements:

```
pip install -r requirements.txt
```

Clone CLIP into this project's repository:

```
git clone https://github.com/openai/CLIP
```

## Acknowledgements

Thanks to Christoph S. from EleutherAI for data processing.

The [clip-retrieval](https://github.com/rom1504/clip-retrieval) repo from rom1504, from which a lot of this code has been inspired from.

## TODO
- [ ] Handle more general input types
- [ ] Batch/Dataset tagging
- [ ] Multi-GPU inference
