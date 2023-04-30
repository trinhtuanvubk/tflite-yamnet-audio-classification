### Setup environment 

###### With conda env
- To create conda environment: 
```bash
conda create --name yamnet-env python=3.8
conda activate yamnet-env 
```

- To install some libs for audio task: 
```bash
sudo apt-get update
sudo apt-get install libsndfile1 -y
sudo apt-get install ffmpeg -y
```

- To install requirements:
```bash
pip install -r requirements.txt
```

###### With Docker container 
- Updated soon 


### Dataset

###### Prepare data
- To download esc50 dataset:
```bash
python3 main.py --scenario  download_data --dataset_name esc-50
```
- Flag:
	- `--dataset_name`: select dataset (`esc-50`, `speech-commands`, `background`)

- To unzip dataset:
```bash
unzip ./dataset-esc50/esc-50.zip
```

- To create category dataset: 
```bash
python3 main.py --scenario  create_category_dataset 
```

- To create splited dataset: 
```bash
python3 main.py --scenario  split_dataset --length_audio 2 --step 0.15
```
- Flag:
	- `--length_audio`: length of the sliding window
	- `--step`: length of the hope of the sliding window

###### Split dataset 
- To train test split:
```bash
python3 main.py --scenario train_test_split --test_data_ratio 0.2 
```
- Flags: 
	- `--test_data_ratio`: ratio of test and train data


### Train and Export

- To run training:
```bash
python3 main.py --scenario train \
--model Yamnet \
--use_custom_dataset \
--train_data_ratio 0.8 \
--epochs 100 \
--batch_size 16 \
--tflite_file_name esc_2s_015_new.tflite \
--save_path ./model_2s_015_new
```
- Flag:
	- `--model`: select model (`Yamnet`, `BrowserFft`)
	- `--train_data_ratio`: ratio of train data and dev data
	- `--epochs`: num epochs 
	- `--batch_size`: num batch size 
	- `--tflite_file_name`: the tflite model name
	- `--save_path`: path to directory contains model 

- To check tflite_model_info: 
```bash
python3 main.py --scenario tflite_model_info
```

### Notes: 
- To modify parameters, go to `scenario/args.py` or through command. 
 
### Build APK file: 

###### Install Android Studio 
- To install, read this tutorial `https://linuxhint.com/install-android-studio-linux-mint-and-ubuntu/`
- Or run the following commands:
```bash
sudo apt update
sudo apt install openjdk-11-jdk
sudo snap install android-studio –classic
```
###### Run default audio_classification app
- To get the repo `https://github.com/tensorflow/examples/tree/master/lite/examples/audio_classification/android`:
```bash
git clone https://github.com/tensorflow/examples.git
cp ./examples/lites/examples/audio_classification ./
cd audio_classification
```
- Start Android Studio, open the project located in `audio_classification/android`, run app with default model: 
```bash
- Select target device menu.
- Click `Run`.
```
###### Copy your model to assets
- To run with yourself model, copy `path/to/esc_model.tflite` to the android app: 
```bash
cp path/to/esc_model.tflite  audio_classification/android/app/src/main/assets/
```

###### Modify params on Android Studio
Go to `/android/app/src/main/java/org/tensorflow/lite/examples/audio/AudioClassificationHelper.kt`. 
- To change model name, at line 136:
```bash
const val YAMNET_MODEL = "path/to/esc_model.tflite"
```
- To change length recordings, (change 1000ms->2000ms), at line 105:
```bash
val lengthInMilliSeconds = ((classifier.requiredInputBufferSize * 1.0f) /
                classifier.requiredTensorAudioFormat.sampleRate) * 2000
```
- To get the result of custom model, change output index from 0->1 (0: result from original yamnet, 1: result from custom yamnet), at line 122: 
```bash
listener.onResult(output[1].categories, inferenceTime)
```

###### Build APK file
- Click `Run` to build app. In the toolbar, to build APK file, click `Build>Build Bunder(s)/APK(s)>Build APK(s)
- Get the APK file at `/audio_classification/android/app/build/intermediates/apk/debug`
- Copy the APK file to the android phone and install. 




