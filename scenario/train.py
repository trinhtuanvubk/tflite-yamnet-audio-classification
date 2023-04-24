import os
import glob
import random
import shutil
# import librosa
# import soundfile as sf
# from IPython.display import Audio
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
import tflite_model_maker as mm
from tflite_model_maker import audio_classifier
from tflite_model_maker.config import ExportFormat

print(f"TensorFlow Version: {tf.__version__}")
print(f"Model Maker Version: {mm.__version__}")

# USE_CUSTOM_DATASET = True

# DATASET_DIR = "./esc_custom_dataset/train"
# TEST_DIR = "./esc_custom_dataset/test"

# DATASET_DIR = "./esc_raw_dataset/train"
# TEST_DIR = "./esc_raw_dataset/test"

# DATASET_DIR = "./esc_custom_dataset_2_015/train"
# TEST_DIR = "./esc_custom_dataset_2_015/test"

# TFLITE_FILENAME = 'esc_2s_015.tflite'
# SAVE_PATH = './models_saving_2s_015'

# batch_size = 16
# epochs = 100


def show_confusion_matrix(confusion, test_labels):
    """Compute confusion matrix and normalize."""
    confusion_normalized = confusion.astype("float") / confusion.sum(axis=1)
    sns.set(rc={'figure.figsize': (6, 6)})
    sns.heatmap(
        confusion_normalized, xticklabels=test_labels, yticklabels=test_labels,
        cmap='Blues', annot=True, fmt='.2f', square=True, cbar=False)
    plt.title("Confusion matrix")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")

def train(args):
    if args.model=="BrowserFft":
        spec = audio_classifier.BrowserFftSpec()
    if args.model=="Yamnet":
        spec = audio_classifier.YamNetSpec(keep_yamnet_and_custom_heads=True)
                    # frame_step=3 * audio_classifier.YamNetSpec.EXPECTED_WAVEFORM_LENGTH,
                    # frame_length=6 * audio_classifier.YamNetSpec.EXPECTED_WAVEFORM_LENGTH)

    if args.use_custom_datset:
        train_data_ratio = args.train_data_ratio
        train_data = audio_classifier.DataLoader.from_folder(
            spec, args.dataset_dir, cache=True)
        train_data, validation_data = train_data.split(train_data_ratio)
        test_data = audio_classifier.DataLoader.from_folder(
            spec, args.test_dir, cache=True)

    model = audio_classifier.create(
        train_data, spec, validation_data, args.batch_size, args.epochs)

    print("=======================")

    model.evaluate(test_data)
    confusion_matrix = model.confusion_matrix(test_data)
    show_confusion_matrix(confusion_matrix.numpy(), test_data.index_to_label)

    print(f'Exporing the model to {args.save_path}')
    model.export(args.save_path, tflite_filename=args.tflite_file_name)
    model.export(args.save_path, export_format=[mm.ExportFormat.SAVED_MODEL, mm.ExportFormat.LABEL])
    
