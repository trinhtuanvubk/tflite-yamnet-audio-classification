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
	- `--length_audio`: length of the splited audio file
	- `--step`: length of the stride  

###### Data loader 
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
--use_custom_datset \
--train_data_ratio 0.8 \
--epochs 100 \
--batch_size 16 \
--tflite_file_name esc_2s_015.tflite \
--save_path ./model_2s_015
```
- Flag:
	- `--model`: select model (`Yamnet`, `BrowserFft`)
	- `--train_data_ratio`: ratio of train data and dev data
	- `--epochs`: num epochs 
	- `--batch_size`: num batch size 
	- `--tflite_file_name`: the tflite model name
	- `--save_path`: path to directory contains model 

### Build APK file: 
- Updated soon

### Notes: 
- To modify parameters, go to `scenario/args.py` or through command. 