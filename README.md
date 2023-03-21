# NCMNet
(CVPR 2023) PyTorch implementation of Paper "Progressive Neighbor Consistency Mining for Correspondence Pruning"

## Requirements

Please use Python 3.6, opencv-contrib-python (3.4.0.12) and Pytorch (>= 1.1.0). Other dependencies should be easily installed through pip or conda.


# Citing NCMNet
If you find the NCMNet code useful, please consider citing:

```bibtex
@inproceedings{liu2023progressive,
  title={Progressive Neighbor Consistency Mining for Correspondence Pruning},
  author={Liu, Xin and Yang, Jufeng},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision.},
  year={2023}
}
```

# Data Processing
The code of this part is partially borrowed from [[OANet](https://github.com/zjhthu/OANet)] [[CNe](https://github.com/vcg-uvic/learned-correspondence-release)]. Please follow their instructions to download the training and testing data.

    bash download_data.sh raw_data raw_data_yfcc.tar.gz 0 8 ## YFCC100M
    tar -xvf raw_data_yfcc.tar.gz

    bash download_data.sh raw_sun3d_test raw_sun3d_test.tar.gz 0 2 ## SUN3D
    tar -xvf raw_sun3d_test.tar.gz
    bash download_data.sh raw_sun3d_train raw_sun3d_train.tar.gz 0 63
    tar -xvf raw_sun3d_train.tar.gz
    
