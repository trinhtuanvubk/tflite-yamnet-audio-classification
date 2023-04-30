import os
import glob
import random


def train_test_split(args):
    # total: 480
    # test: 96
    dirs = glob.glob(os.path.join(args.dataset_dir, '*/'))
    for dir in dirs:
        print(dir)
        files = glob.glob(os.path.join(dir, '*.wav'))
        print(len(files))
        test_count = round(len(files) * args.test_data_ratio)
        random.seed(42)
        random.shuffle(files)
        # Move test samples:
        for file in files[:test_count]:
            class_dir = os.path.basename(os.path.normpath(dir))
            os.makedirs(os.path.join(args.test_dir, class_dir), exist_ok=True)
            os.rename(file, os.path.join(args.test_dir,
                      class_dir, os.path.basename(file)))
        print('Moved', test_count, 'audio files from', class_dir)
