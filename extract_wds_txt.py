import os
import webdataset as wds

from argparse import ArgumentParser
from torch.utils.data import DataLoader
from pathlib import Path


def web_dataset_helper(path):
    """
    https://github.com/tgisaturday/dalle-lightning/blob/master/pl_dalle/loader.py
    """
    if Path(path).is_dir():
        DATASET = [str(p) for p in Path(path).glob("**/*") if ".tar" in str(p).lower()] # .name
        assert len(DATASET) > 0, 'The directory ({}) does not contain any WebDataset/.tar files.'.format(path)
        print('Found {} WebDataset .tar(.gz) file(s) under given path {}!'.format(len(DATASET), path))
    elif ('http://' in path.lower()) | ('https://' in path.lower()):
        DATASET = f"pipe:curl -L -s {path} || true"
        print('Found {} http(s) link under given path!'.format(len(DATASET), path))
    elif 'gs://' in path.lower():
        DATASET = f"pipe:gsutil cat {path} || true"
        print('Found {} GCS link under given path!'.format(len(DATASET), path))
    elif '.tar' in path:
        DATASET = path
        print('Found WebDataset .tar(.gz) file under given path {}!'.format(path))
    else:
        raise Exception('No folder, no .tar(.gz) and no url pointing to tar files provided under {}.'.format(path))
    return DATASET


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--input_dir', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--wds_key', type=str, default='txt')
    parser.add_argument('--prefix', type=str, default='out')
    parser.add_argument('--num_prepro_workers', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=256)
    args = parser.parse_args()

    DATASET = web_dataset_helper(args.input_dir)
    mapping = {args.wds_key: lambda s: s.decode('utf-8').strip()}
    dataset = wds.WebDataset(DATASET).map_dict(**mapping).to_tuple(args.wds_key)
    dataloader = DataLoader(dataset,
                            num_workers=args.num_prepro_workers,
                            batch_size=args.batch_size)
    output_str = ''
    for batch in dataloader:
        for text in batch[0]:
            output_str += f'{text}\n'
    p = Path(os.path.join(args.output_dir, f'{args.prefix}.txt'))
    p.write_text(output_str)