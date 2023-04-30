import os
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
import tflite_model_maker as mm
from tflite_model_maker import audio_classifier
from tflite_model_maker.config import ExportFormat

print(f"TensorFlow Version: {tf.__version__}")
print(f"Model Maker Version: {mm.__version__}")


def show_confusion_matrix(args, confusion, test_labels):
    """Compute confusion matrix and normalize."""
    file_path = os.path.join(args.save_path, "confusion_matrix.png")

    confusion_normalized = confusion.astype("float") / confusion.sum(axis=1)
    sns.set(rc={'figure.figsize': (50, 50)})
    sns.heatmap(
        confusion_normalized, xticklabels=test_labels, yticklabels=test_labels,
        cmap='Blues', annot=True, fmt='.2f', square=True, cbar=False)
    plt.title("Confusion matrix")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.savefig(file_path)


def train(args):
    if args.model == "BrowserFft":
        spec = audio_classifier.BrowserFftSpec()
    if args.model == "Yamnet":
        spec = audio_classifier.YamNetSpec(keep_yamnet_and_custom_heads=True,
                                           frame_step=audio_classifier.YamNetSpec.EXPECTED_WAVEFORM_LENGTH // 2,
                                           frame_length=args.length_audio * audio_classifier.YamNetSpec.EXPECTED_WAVEFORM_LENGTH)

    if args.use_custom_dataset:
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
    show_confusion_matrix(args, confusion_matrix.numpy(),
                          test_data.index_to_label)

    print(f'Exporing the model to {args.save_path}')
    model.export(args.save_path, tflite_filename=args.tflite_file_name)
    model.export(args.save_path, export_format=[
                 mm.ExportFormat.SAVED_MODEL, mm.ExportFormat.LABEL])
