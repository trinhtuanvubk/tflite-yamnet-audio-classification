import argparse


def get_args():
    # create args parser
    parser = argparse.ArgumentParser(description='Sound_Classification')

    # params for downloading dataset
    parser.add_argument('--dataset_name', type=str, default='esc-50')
    parser.add_argument('--metadata_path', type=str,
                        default='./datasets-esc50/ESC-50-master/meta/esc50.csv')
    parser.add_argument('--data_path', type=str,
                        default='./datasets-esc50/ESC-50-master/audio')
    parser.add_argument('--raw_dataset_path', type=str,
                        default='./datasets/esc_raw_dataset/full')

    # params for preparing dataset
    parser.add_argument('--splited_dataset_path', type=str,
                        default='./datasets/esc_custom_dataset/train')
    parser.add_argument('--length_audio', type=int, default=2)
    parser.add_argument('--step', type=float, default=0.15)
    parser.add_argument('--dataset_dir', type=str,
                        default='./datasets/esc_custom_dataset/train')
    parser.add_argument('--test_dir', type=str,
                        default='./datasets/esc_custom_dataset/test')
    parser.add_argument('--test_data_ratio', type=float, default=0.2)

    # params for training model
    parser.add_argument('--use_custom_dataset', action='store_true')
    parser.add_argument('--model', type=str, default='Yamnet')
    parser.add_argument('--train_data_ratio', type=float, default=0.8)
    parser.add_argument('--tflite_file_name', type=str,
                        default='esc_2s_015.tflite')
    parser.add_argument('--save_path', type=str, default='./model_2s_015')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=100)

    # params for scenario
    parser.add_argument('--scenario', type=str, default='train')

    args = parser.parse_args()
    return args
