# COMPUTER
Implementation for the paper "Unified Compositional Query Machine with Multimodal Consistency for Video-based Human Activity Recognition" (Tuyen et al., BMVC 2024)

## Installation
The code was developed and tested on Ubuntu 18.04 LTS using Python 3.9.13

Before running the code, please install Anaconda to create a Python environment.

Assuming Anaconda is already installed, use the following command to install the required dependencies:

```
conda env create -f environment.yml
conda activate computer
```
## Data preparation and features extraction 
- Raw data: Follow the instructions in the [Slowfast repo](https://github.com/facebookresearch/SlowFast/blob/main/slowfast/datasets/DATASET.md) to download the dataset and perform the necessary preprocessing steps. After completing these steps, the AVA dataset should have the following structure:
    ```
    ava
    |_ frames
    |  |_ [video name 0]
    |  |  |_ [video name 0]_000001.jpg
    |  |  |_ [video name 0]_000002.jpg
    |  |  |_ ...
    |  |_ [video name 1]
    |     |_ [video name 1]_000001.jpg
    |     |_ [video name 1]_000002.jpg
    |     |_ ...
    |_ frame_lists
    |  |_ train.csv
    |  |_ val.csv
    |_ annotations
    |_ [official AVA annotation files]
    |_ ava_train_predicted_boxes.csv
    |_ ava_val_predicted_boxes.csv
    ```
- Features extraction: 
    - Download the backbone model at [here](https://drive.google.com/file/d/1gH4La9w_HSOuTIHGHoY_5OFZLrcEUR16/view).
    - Execute the following commands to extract pre-computed clip selections. Choose either `train` or `test` data by using the `--mode` flag:

        ```bash
            python extract_feature/clip_selection.py --mode train --temp_path path/to/save/clip/selection --backbone_model path/to/backbone/model
        ```
        Then set the path for the extracted pre-computed clip in the config file you are using (e.g., `Computer_MViT_S_K400.yaml`) by modifying `COMPUTER.CLIP_SELECTION.`
    - Run the following command to extract human and context features from the visual backbone,  choose either `train` or `test` data by using the `--mode` flag:
        ```bash
            python extract_feature/feature_extraction.py --mode train --feature_path path/to/save/feature --backbone_model path/to/backbone/model
        ```
        Then set the path for the extracted feature in the config file you are using (e.g., `Computer_MViT_S_K400.yaml`) by modifying `COMPUTER.FEATURE_EXTRACTION_TRAIN` and `COMPUTER.FEATURE_EXTRACTION_TEST`.


    - Run the following command to extract skeleton data,  change the paht `video_path` to the directory of keyframe for train or val set:
        ```bash
            python extract_feature/extract_key_point.py
        ```
        Then set the path for the extracted feature in the config file you are using (e.g., `Computer_MViT_S_K400.yaml`) by modifying `COMPUTER.SKELETON_TRAIN` and `COMPUTER.SKELETON_VAL`.
## Running the method
Setup environment by running `setup.py` to add the project path to the PYTHONPATH
```bash
python setup.py build develop
```

#### Training
- Download the MViTv2-S pretrained on Kinetic at [here](https://drive.google.com/file/d/1titTOtlYjsdcm_ZrzzvReOwPXT6fPuPR/view), then set the path for this at `TRAIN.CHECKPOINT_FILE_PATH` in the config file that you used (e.g `Computer_MViT_S_K400.yaml`)
- Execute the following command to train COMPUTER
    ```
        python tools/run_net.py --cfg configs/Computer_MViT_S_K400.yaml
    ```
#### Testing
- Run the following command to test COMPUTER:
    ```
        python tools/run_net.py --cfg configs/Computer_MViT_S_K400.yaml TRAIN.ENABLE False TEST.CHECKPOINT_FILE_PATH path/to/the/pretrain
    ```

## Acknowledgements
This project is built upon the code bases [PySlowfast](https://github.com/facebookresearch/SlowFast). We would like to thank the authors for releasing their code.
