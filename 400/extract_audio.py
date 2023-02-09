import argparse
import os 
import sys 
import glob
import moviepy.editor as mp

from joblib import delayed, Parallel 
from tqdm import tqdm 


failed_videos = []


def extract_audio(v_path, f_root):
    '''v_path: single video path;
       f_root: root to store frames'''
    global failed_videos
    
    clip_path = os.path.join(
        f_root,
        '/'.join(''.join(v_path.split('.')[:-1]).split('/')[-2:]),
        'audio.wav'
    )
    try:
        my_clip = mp.VideoFileClip(v_path)
        my_clip.audio.write_audiofile(clip_path, verbose=False, logger=None)
    except Exception:
        failed_videos.append(v_path)
    

def main(v_root, f_root, splits, num_workers=32):
    global failed_videos
    print('extracting Kinetics400 ... ')
    for split in splits:
        v_root_real = os.path.join(v_root, split)
        f_root_real = os.path.join(f_root, split)
        if not os.path.exists(v_root_real):
            print('Wrong v_root'); sys.exit()
        else:
            print('Extract to: \nframe: %s' % f_root_real)
        
        if not os.path.exists(f_root_real): os.makedirs(f_root_real)
        v_act_root = glob.glob(os.path.join(v_root_real, '*/'))
        v_act_root = sorted(v_act_root)

        failed_videos = []
        # if resume, remember to delete the last video folder
        for i, j in tqdm(enumerate(v_act_root), total=len(v_act_root)):
            target_class = v_act_root[i].split('/')[-2]
            with open('finished.txt', 'r') as f:
                finished_classes = f.read().splitlines()
            if target_class in finished_classes:
                continue
            
            v_paths = glob.glob(os.path.join(j, '*.mp4'))
            v_paths = sorted(v_paths)
            Parallel(num_workers)(delayed(extract_audio)(p, f_root_real) for p in v_paths)
            
            with open('finished.txt', 'a') as f:
                f.write(target_class + '\n')
        
        print('Here is a list of the videos which failed to extract audio: \n' + '\n'.join(failed_videos))


if __name__ == '__main__':
    # v_root is the video source path, f_root is where to store frames
    parser = argparse.ArgumentParser("Extracts frames from Kinetrics videos")
    parser.add_argument('-s', '--split', type=str, default='val', choices=('train', 'test', 'val', 'all'),
                        help="choose dataset split ('train', 'test', 'val', or 'all')")
    parser.add_argument('-w', '--num-workers', type=int, default=32, 
                        help="Set number of multiprocessing workers")
    opt = parser.parse_args()
    
    opt.num_workers = min(opt.num_workers, os.cpu_count())
    opt.splits = [opt.split] if opt.split != 'all' else ['train', 'test', 'val']
    
    main(v_root='/lisa_migration/project/prjs_papostolos/kinetics400/videos',
         f_root='/scratch-shared/papostolos/kinetics400/frames',
         splits=opt.splits,
         num_workers=opt.num_workers,
    )