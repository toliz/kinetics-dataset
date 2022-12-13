import os
import os.path as osp
import argparse
from multiprocessing import Pool
from tqdm import tqdm
import tarfile
import pathlib


DATASET_PATH = '/project/prjs_papostolos/kinetics400'
TARGZ_PATH = '/project/prjs_papostolos/kinetics400_targz/'

global split


def extract_class(class_name):
    global split
    
    pathlib.Path(f'{DATASET_PATH}/{split}/{class_name}').mkdir(parents=True, exist_ok=True)
  
    file = tarfile.open(f'{TARGZ_PATH}/{split}/{class_name}.tar.gz')
    file.extractall(f'{DATASET_PATH}/{split}/{class_name}')
    file.close()


def main(opt):
    global split
    
    opt.num_workers = min(opt.num_workers, os.cpu_count())
    opt.splits = [opt.split] if opt.split != 'all' else ['train', 'test', 'validate']
    
    for split in opt.splits:
        class_names = [file[:-7] for file in os.listdir(f'{TARGZ_PATH}/{split}') if file.endswith('.tar.gz')]
        
        pool = Pool(opt.num_workers)
        for _ in tqdm(pool.imap_unordered(extract_class, class_names), total=len(class_names)):
            pass
        pool.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Kinetics dataset downloader")
    parser.add_argument('-s', '--split', type=str, default='val', choices=('train', 'test', 'val', 'all'),
                        help="choose dataset split ('train', 'test', 'val', or 'all')")
    parser.add_argument('-w', '--num-workers', type=int, default=32, help="Set number of multiprocessing num_workers")
    opt = parser.parse_args()
    
    main(opt)
