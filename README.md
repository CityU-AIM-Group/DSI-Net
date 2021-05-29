# DSI-Net

This repository is an official PyTorch implementation of the paper [**"DSI-Net: Deep Synergistic Interaction Network for Joint Classification and Segmentation with Endoscope Images"**](https://ieeexplore.ieee.org/document/9440441), TMI 2021.

<div align=center><img width="700" src=/Figs/Framework.png></div>


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
 
## Citation
```
@ARTICLE{9440441,
  author={Zhu, Meilu and Chen, Zhen and Yuan, Yixuan},
  journal={IEEE Transactions on Medical Imaging}, 
  title={DSI-Net: Deep Synergistic Interaction Network for Joint Classification and Segmentation with Endoscope Images}, 
  year={2021},
  doi={10.1109/TMI.2021.3083586}}
```
## Contact

  Meilu Zhu (meiluzhu2@cityu.edu.hk)
