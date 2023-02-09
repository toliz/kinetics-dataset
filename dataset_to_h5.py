import argparse
import h5py
import json
import librosa
import numpy as np
import os
from skimage.io import imread
from scipy.io import wavfile
from scipy.signal import resample
from pathlib import Path
from multiprocessing import Pool
from tqdm import tqdm


def center_crop(img, size=(256, 256)):
    """
    Center crop an image to a given size.
    
    Args:
        img: A (H, W, 3) numpy array containg the image to be cropped
        size: A tuple containing the desired size of the cropped image
        
    Returns:
        A (size[0], size[1], 3) numpy array containing the cropped image
    """
    h, w = img.shape[:2]
    th, tw = size
    i = int(round((h - th) / 2.))
    j = int(round((w - tw) / 2.))
    return img[i:i+th, j:j+tw]


def read_frames(path):
    """
    Concat the frames of a video into a numpy array.
    
    Args:
        path: the directory containing the frames of a video
        
    Returns:
        A (n_frames, 3, 256, 256) numpy array containing the frames of the video
    """
    n_frames = len(os.listdir(path))
    frames = np.empty((n_frames, 3, 256, 256), dtype=np.uint8)
    for i in range(n_frames):
        img = imread(os.path.join(path, f'image_{i+1:05d}.jpg'))
        frames[i] = center_crop(img).transpose(2, 0, 1)
        
    return frames


def read_audio(path, sr=16000):
    """
    Read the audio of a video into a numpy array.
    
    Args:
        path: the directory containing the audio of a video
        
    Returns:
        A (sr*duration, ) numpy array containing the audio of the video
    """
    try:
        orig_sr, audio = wavfile.read(os.path.join(path, 'audio.wav'))
        audio = audio.mean(axis=1)                              # single channel
        audio = resample(audio, int(sr*audio.shape[0]/orig_sr)) # resample
    except Exception as e:
        audio = np.empty((0, ))
        
    return audio


def main(frame_dir, audio_dir, h5py_dir, splits, num_workers):
    for split in splits:
        # Process each class
        for cls in tqdm(sorted(os.listdir(os.path.join(frame_dir, split)))):
            # Skip class if already processed
            if os.path.exists(os.path.join(h5py_dir, split, f'{cls}.h5')):
                continue
            
            # Read frames
            samples = os.listdir(os.path.join(frame_dir, split, cls))
            video_paths = [os.path.join(frame_dir, split, cls, sample) for sample in samples]
            audio_paths = [os.path.join(audio_dir, split, cls, sample) for sample in samples]
            videos = Pool(num_workers).map(read_frames, video_paths)
            audios = Pool(num_workers).map(read_audio, audio_paths)
            
            # Keep only videos with at least 3 frames and audio
            idx = [i for i, (video, audio) in enumerate(zip(videos, audios)) if video.shape[0] >= 3 and audio.shape[0] >= 3*16000]
            videos = [videos[i] for i in idx]
            audios = [audios[i] for i in idx]
            video_lens = [video.shape[0] for video in videos]
            audio_lens = [audio.shape[0] for audio in audios]
            
            Path(os.path.join(h5py_dir, split)).mkdir(parents=True, exist_ok=True)
            with h5py.File(os.path.join(h5py_dir, split, f'{cls}.h5'), 'w') as f:
                f.create_dataset('frames', data=np.concatenate(videos), dtype=np.uint8)
                f.create_dataset('audios', data=np.concatenate(audios), dtype=np.float16)
                f.create_dataset('frames_idx', data=np.cumsum([0] + video_lens), dtype=np.uint64)
                f.create_dataset('audios_idx', data=np.cumsum([0] + audio_lens), dtype=np.uint64)
                f.create_dataset('yt_idx', data=np.char.encode(samples, "utf-8"), dtype='S25')

        # Store dataset info - necessary for the dataloader                
        class_names, class_indices = [], []
        for cls in tqdm(sorted(os.listdir(os.path.join(frame_dir, split)))):
            with h5py.File(os.path.join(h5py_dir, split, f'{cls}.h5'), 'r') as f:
                try:
                    class_names.append(cls)
                    class_indices.append(len(f['frames_idx'])-1)
                except Exception:
                    print(cls)
        json.dump(
            {'names': class_names, 'indices': (np.cumsum(class_indices)-1).tolist()},
            open(os.path.join(h5py_dir, split, 'info.json'), 'w')
        )
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Converts a collection of .jpgs and .wav files into hdf5.")
    parser.add_argument('-s', '--split', type=str, default='val', choices=('train', 'test', 'val', 'all'),
                        help="choose dataset split ('train', 'test', 'val', or 'all')")
    parser.add_argument('-w', '--num-workers', type=int, default=32, 
                        help="Set number of multiprocessing workers")
    opt = parser.parse_args()
    
    opt.num_workers = min(opt.num_workers, os.cpu_count())
    opt.splits = [opt.split] if opt.split != 'all' else ['train', 'test', 'val']
    
    main(
        frame_dir='/lisa_migration/project/prjs_papostolos/kinetics400/frames',
        audio_dir='/scratch-shared/papostolos/kinetics400/frames',
        h5py_dir='/scratch-shared/papostolos/',
        splits=opt.splits,
        num_workers=opt.num_workers,
    )