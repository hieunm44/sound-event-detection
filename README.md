# Sound Event Detection
<div align="center">
<img src="https://dcase.community/images/tasks/challenge2017/task3_overview.png" height=400"/>
</div>

## Overview
This repo implements the [DCASE 2017 Challenge - Task 3: Sound event detection in real life audio](https://dcase.community/challenge2017/task-sound-event-detection-in-real-life-audio), using the [Convolutional Recurrent Neural Network (CRNN)](https://arxiv.org/abs/1706.02291). This task evaluates performance of the sound event detection systems in multisource conditions similar to our everyday life, where the sound sources are rarely heard in isolation.

## Built With
<div align="center">
<a href="https://librosa.org/">
  <img src="https://librosa.org/images/librosa_logo_text.png" height=40 hspace=10/>
</a>
<a href="https://www.tensorflow.org/">
  <img src="https://www.gstatic.com/devrel-devsite/prod/vdc54107fd8beee9a25bbc52caca7c5cd8d6bde91b94b693cf51910bd553c2293/tensorflow/images/lockup.svg" height=40 hspace=10/>
</a>
<a href="https://keras.io/">
  <img src="https://keras.io/img/logo.png" height=40/>
</a>
</div>

## Usage
1. Clone the repo
   ```sh
   git clone https://github.com/hieunm44/sound-event-detection.git
   cd sound-event-detection
   ```
2. Install necessary packages
   ```sh
   pip install -r requirements.txt
   ```
3. Download the dataset: https://zenodo.org/record/814831 \
   Extract the downloaded files and we should have two folders: `audio/street` and `evaluation_setup` containing audio files and label files, respectively.
4. Extract audio features
   ```sh
   python3 feature_extraction.py
   ```
   The extracted features will be saved in the folder `feat` as `npz` files.
5. Train the model
   ```sh
   python3 main.py
   ```
   The trained models will be saved in the folder `models`.


