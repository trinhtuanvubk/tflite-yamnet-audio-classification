import os
import glob
import random
import shutil
import csv
import pandas as pd
import librosa
import soundfile as sf
import numpy as np
import tensorflow as tf

from scenario.args import get_args


def test_dataset_size(recording_list):
    assert len(recording_list) == 2000


def test_recordings(recording_list):
    for recording in tqdm.tqdm(recording_list):
        signal, rate = librosa.load('audio/' + recording, sr=None, mono=False)

        assert rate == 44100
        assert len(signal.shape) == 1  # mono
        assert len(signal) == 220500  # 5 seconds
        assert np.max(signal) > 0
        assert np.min(signal) < 0
        assert np.abs(np.mean(signal)) < 0.2  # rough DC offset check


def download_data(args):
    if args.dataset_name == "esc-50":
        tf.keras.utils.get_file('esc-50.zip',
                                'https://github.com/karoldvl/ESC-50/archive/master.zip',
                                cache_dir='./',
                                cache_subdir='datasets-esc50',
                                extract=True)
    if args.dataset_name == "speech-commands":
        tf.keras.utils.get_file('speech_commands_v0.01.tar.gz',
                                'http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz',
                                cache_dir='./',
                                cache_subdir='dataset-commands',
                                extract=True)
    if args.dataset_name == "backgroud":
        tf.keras.utils.get_file('background_audio.zip',
                                'https://storage.googleapis.com/download.tensorflow.org/models/tflite/sound_classification/background_audio.zip',
                                cache_dir='./',
                                cache_subdir='dataset-background',
                                extract=True)


def view_metadata(args):
    pd_data = pd.read_csv(args.metadata_path)
    pd_data.head()


def create_category_dataset(args):
    with open(args.metadata_path) as file:
        data = csv.reader(file, delimiter=",")
        next(data)
        for idx, row in enumerate(data):
            # ['filename', 'fold', 'target', 'category', 'esc10', 'src_file', 'take']
            filename = row[0]
            category = row[3]
            category_path = os.path.join(args.raw_dataset_path, category)
            # if not os.path.isdir(category_path):
            os.makedirs(category_path, exist_ok=True)
            input_path = os.path.join(args.data_path, filename)
            output_path = os.path.join(category_path, filename)
            shutil.copy2(input_path, output_path)


def split_dataset(args):
    categories = glob.glob(os.path.join(args.raw_dataset_path, '*'))
    print(categories)
    # output_categories = os.path.join(splited_dataset_path, categories)
    for category in categories:
        category_name = os.path.basename(os.path.normpath(category))
        output_category_path = os.path.join(
            args.splited_dataset_path, category_name)
        print(output_category_path)
        # if not os.path.isdir(output_category_path):
        os.makedirs(output_category_path, exist_ok=True)
        files = glob.glob(os.path.join(category, '*.wav'))
        for file in files:
            print(file)
            # data, sr = librosa.load(file_path, sr=44100)
            file_name = os.path.basename(os.path.normpath(file))
            file_base_name = os.path.splitext(file_name)[0]
            # with open(file, 'rb') as f: 
            #     data, sr = sf.read(f, channels=1, samplerate=44100, subtype="FLOAT")
            data, sr = librosa.load(file, sr=44100)
            data = librosa.to_mono(data)
            frame_length = int(sr*args.length_audio)
            hop_length = int(sr*args.step)
            for idx, frame in enumerate(range(0, len(data)-frame_length, hop_length)):
                splited_data = data[frame:frame+frame_length]
                splited_data = splited_data.T
                output_file_path = os.path.join(output_category_path, file_name + str(idx) + '.wav')
                sf.write(output_file_path, splited_data, sr, format='WAV', subtype='PCM_16')


# if __name__ == "__main__":
#     # download_data(dataset_name="esc-50")
#     # view_metadata()
#     create_category_dataset()
#     split_dataset(raw_dataset_path="./esc_raw_dataset/full",
#                   splited_dataset_path="./esc_custom_dataset_2_015/train",
#                   length_audio=2,
#                   stride=0.15)