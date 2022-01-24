# DHDRNet
High Dynamic Range Imaging for Dynamic Scenes  with Large-Scale Motions and Severe Saturation
By Xiao Tan, Huaian Chen, Kai Xu, Chunmei Xu, Yi Jin, Changan Zhu

### Highlights
- **self-guided attention**: reduces the influence of the saturated regions in the alignment and fusion processes
- **a pyramidal deformable module**: effectively remove ghosting artifacts

## Dependencies and Installation

- Python 3 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux))
- [PyTorch >= 1.1](https://pytorch.org/)
- NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)
- [Deformable Convolution](https://arxiv.org/abs/1703.06211). We use [mmdetection](https://github.com/open-mmlab/mmdetection)'s dcn implementation. Please first compile it.
  ```
  cd ./dcn
  python setup.py develop
  ```

## Dataset Preparation Using MATLAB
We use datasets in h5 format for faster IO speed. 
Please unzip the [training and test datasets](https://cseweb.ucsd.edu/~viscomp/projects/SIG17HDR/) into ./dataset_select/Data
  ```
  cd ./dataset_select
  run PrepareData.m
  ```

## Train
  ```
  python train.py
  ```

## Test
  ```
  python test.py
  ```
  
## Citation
If you find this code useful in your research, please consider citing:
  ```
  @ARTICLE{DHDRNet,
  author={Tan, Xiao and Chen, Huaian and Xu, Kai and Xu, Chunmei and Jin, Yi and Zhu, Changan and Zheng, Jinjin},
  journal={IEEE Transactions on Instrumentation and Measurement}, 
  title={High Dynamic Range Imaging for Dynamic Scenes with Large-Scale Motions and Severe Saturation}, 
  year={2022},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TIM.2022.3144205}
  }
  ```
