import argparse
import os 
import sys 
import glob
import cv2
import matplotlib.pyplot as plt

from joblib import delayed, Parallel 
from tqdm import tqdm 


plt.switch_backend('agg')


def extract_video_opencv(v_path, f_root, fps=1, dim=240):
    '''v_path: single video path;
       f_root: root to store frames'''
    v_class = v_path.split('/')[-2]
    v_name = os.path.basename(v_path)[0:-4]
    out_dir = os.path.join(f_root, v_class, v_name)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    vidcap = cv2.VideoCapture(v_path)
    nb_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float
    height = vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT) # float
    if (width == 0) or (height==0): 
        print(v_path, 'not successfully loaded, drop ..'); return
    new_dim = resize_dim(width, height, dim)

    success, image = vidcap.read()
    count = 1
    while success:
        image = cv2.resize(image, new_dim, interpolation = cv2.INTER_LINEAR)
        cv2.imwrite(os.path.join(out_dir, 'image_%05d.jpg' % count), image,
                    [cv2.IMWRITE_JPEG_QUALITY, 80])# quality from 0-100, 95 is default, high is good
        vidcap.set(cv2.CAP_PROP_POS_MSEC,int(count*1000/fps))
        success, image = vidcap.read()
        count += 1
    if nb_frames > count * 1000 / fps:
        print('/'.join(out_dir.split('/')[-2::]), 'NOT extracted successfully: %df/%df' % (count, nb_frames))
    vidcap.release()


def resize_dim(w, h, target):
    '''resize (w, h), such that the smaller side is target, keep the aspect ratio'''
    if w >= h:
        return (int(target * w / h), int(target))
    else:
        return (int(target), int(target * h / w)) 


def main(v_root, f_root, splits, num_workers=32, fps=1, dim=150):
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

        # if resume, remember to delete the last video folder
        for i, j in tqdm(enumerate(v_act_root), total=len(v_act_root)):
            v_paths = glob.glob(os.path.join(j, '*.mp4'))
            v_paths = sorted(v_paths)
            # for resume:
            v_class = j.split('/')[-2]
            out_dir = os.path.join(f_root_real, v_class)
            if os.path.exists(out_dir): continue
            # dim = 150 (crop to 128 later) or 256 (crop to 224 later)
            Parallel(num_workers)(delayed(extract_video_opencv)(p, f_root_real, fps=fps, dim=dim) for p in v_paths)


if __name__ == '__main__':
    # v_root is the video source path, f_root is where to store frames
    parser = argparse.ArgumentParser("Extracts frames from Kinetrics videos")
    parser.add_argument('-s', '--split', type=str, default='val', choices=('train', 'test', 'val', 'all'),
                        help="choose dataset split ('train', 'test', 'val', or 'all')")
    parser.add_argument('-w', '--num-workers', type=int, default=32, 
                        help="Set number of multiprocessing workers")
    parser.add_argument('--fps', type=int, default=1,
                        help='Frames to extract per second')
    parser.add_argument('--dim', type=int, default=256, 
                        help="Dimensionality of the extracted frames")
    opt = parser.parse_args()
    
    opt.num_workers = min(opt.num_workers, os.cpu_count())
    opt.splits = [opt.split] if opt.split != 'all' else ['train', 'test', 'val']
    
    main(v_root='/home/papostolos/datasets/kinetics400/videos',
         f_root='/home/papostolos/datasets/kinetics400/frames',
         splits=opt.splits,
         num_workers=opt.num_workers,
         fps=opt.fps,
         dim=opt.dim
    )