import argparse
import os
import pandas as pd
from multiprocessing import Pool
import shutil
from tqdm import tqdm
import pathlib


DATASET_PATH = '/home/papostolos/datasets/kinetics400/'
global split


def mv(youtube_id, time_start, time_end, label):
    global split
    try:
        shutil.move(
            src=os.path.join(DATASET_PATH, split, f'{youtube_id}_{time_start:06d}_{time_end:06d}.mp4'),
            dst=os.path.join(DATASET_PATH, 'videos', split, label, f'{youtube_id}_{time_start:06d}_{time_end:06d}.mp4')
        )
    except Exception:
        pass
    
    
def mv_wrapper(args):
    mv(*args)
    

def main(opt):
    global split
    
    opt.num_workers = min(opt.num_workers, os.cpu_count())
    opt.splits = [opt.split] if opt.split != 'all' else ['train', 'test', 'val']
    
    for split in opt.splits:
        annotations = pd.read_csv(open(f'{DATASET_PATH}/annotations/{split}.csv'))
        for orig, sub in [(' ', '_'), ('(', ''), (')', ''), ('\'', '')]:
            annotations.label = annotations.label.str.replace(orig, sub, regex=False)
        
        # create a directory for each label in the dataset
        labels = annotations.label.drop_duplicates().to_list()
        for label in labels:
            pathlib.Path(DATASET_PATH, 'videos', split, label).mkdir(parents=True, exist_ok=True)
            
        # start moving files to their corresponding directories
        pool = Pool(opt.num_workers)
        args = zip(annotations.youtube_id, annotations.time_start, annotations.time_end, annotations.label)    
        for _ in tqdm(pool.imap_unordered(mv_wrapper, args), total=len(annotations)):
            pass
        pool.close()
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Organizes downloaded videos into subfolders according to their labels")
    parser.add_argument('-s', '--split', type=str, default='val', choices=('train', 'test', 'val', 'all'),
                        help="choose dataset split ('train', 'test', 'val', or 'all')")
    parser.add_argument('-w', '--num-workers', type=int, default=32, 
                        help="Set number of multiprocessing workers")
    opt = parser.parse_args()
    
    main(opt)
