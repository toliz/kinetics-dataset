import argparse
import os 
import shutil

from tqdm import tqdm 
    

def main(root, splits):
    for split in splits:
        directory = os.path.join(root, split)
        n_corrupted = 0
        for target_class in tqdm(os.listdir(directory)):
            for video in os.listdir(os.path.join(directory, target_class)):
                contents = os.listdir(os.path.join(directory, target_class, video))
                if 'audio.wav' not in contents or 'image_00001.jpg' not in contents:
                    n_corrupted += 1
                    shutil.rmtree(os.path.join(directory, target_class, video))
        
        print(f'Split {split} had {n_corrupted} corrupted videos.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Removes videos for which frames or audio could not be extracted")
    parser.add_argument('-s', '--split', type=str, default='val', choices=('train', 'test', 'val', 'all'),
                        help="choose dataset split ('train', 'test', 'val', or 'all')")
    opt = parser.parse_args()
    opt.splits = [opt.split] if opt.split != 'all' else ['train', 'test', 'val']
    
    main('/scratch-shared/papostolos/kinetics400/frames', opt.splits)