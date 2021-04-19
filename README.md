# DSI-Net

This repository is an official PyTorch implementation of the paper **"DSI-Net: Deep Synergistic Interaction Network for Joint Classification and Segmentation with Endoscope Images"**, under review at **TMI 2021**.

<div align=center><img width="900" src=/Figs/Framework.png></div>


## Dependencies
* Python 3.6
* PyTorch >= 1.3.0
* numpy
* apex
* sklearn
* matplotlib
* PIL

## Usage
* Training DSI-Net
   ```python
   python train_DSI_Net.py --gpus 0 --K 100 --alpha 0.05 --image_list 'data/WCE/WCE_Dataset_image_list.pkl'
   ```
* Testing DSI-Net

  Download our trained model <a href="https:#" target="_blank">here</a> and put it in ```\checkpoints```. Please crop the black margin before test.
   ```python
   python test_DSI_Net.py --img_path xx --model xxx
   ``` 
  
 
