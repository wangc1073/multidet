This repository provides an implementation of the multidet model.

**Paper:**
*A Transformer-Based Method of Multienergy Load Forecasting in Integrated Energy System*, 
IEEE Transactions on Smart Grid, Vol. 13, No. 4, July 2022
DOI: 10.1109/TSG.2022.3166600
Paper link: https://ieeexplore.ieee.org/document/9756020

## Data Preparation

Please note that the `ReadData` function is provided as an interface and should be customized according to your own data format and preprocessing pipeline.

The multi-energy load dataset used in the paper can be downloaded from:

https://cm.asu.edu/

After downloading the dataset, please preprocess the data accordingly and modify the `ReadData` function before running the demo.

## Citation

If you find this repository or our work useful for your research, please consider citing the following paper:

```bibtex
@article{wang2022transformer,
  title={A Transformer-Based Method of Multienergy Load Forecasting in Integrated Energy System},
  author={Wang, Chen and Wang, Ying and Ding, Zhetong and Zheng, Tao and Hu, Jian-Gen and Zhang, Ji},
  journal={IEEE Transactions on Smart Grid},
  volume={13},
  number={4},
  pages={2703--2714},
  year={2022},
  publisher={IEEE}
}
```
