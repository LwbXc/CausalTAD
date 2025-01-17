# CausalTAD
Code for ICDE'24 paper: "CausalTAD: Causal Implicit Generative Model for Debiased Online Trajectory Anomaly Detection" and several other trajectory anomaly detection models: SAE, VSAE, GM-VSAE.

# Dataset
Please unzip files `./datasets/xian.rar` and `./datasets/porto.rar` first.
There are datasets for two cities (Xi'an and Porto) in the `./datasets` directory, including `train-grid.pkl` for training, `valid-normal-grid.pkl` and `valid-abnormal-grid.pkl` for validation, and `test-grid.pkl` and `alpha_i_distance_j.pkl` ($i \in [0.1, 0.2, 0.3]$ denotes the ratio of trajectories that is perturbed and $j \in [1, 2, 3]$ denotes level of perturbation). More details about the generation of outliers can be found in the `./datasets/xian/generate_outlier.py` or `./datasets/porto/generate_outlier.py`.

# Usage
For instance, to train CausalTAD on trajectories of Xi'an and Porto, and evaluate the trained models, you can run following commands:
```
cd CausalTAD
python main.py
```
The usage of other models is similar.

# Acknowledgements
The implementation of GM-VSAE is based on the code from [this repository](https://github.com/chwang0721/GM-VSAE), I simplified the code for pretraining and training Gaussian Mixture Model to make the logic of model optimization more straightforward, and I modified the code of model inference based on my own understanding.

# Citation
Please cite our paper if our paper or code is helpful.

```
@inproceedings{Li0GCJZZFB24,
  author       = {Wenbin Li and
                  Di Yao and
                  Chang Gong and
                  Xiaokai Chu and
                  Quanliang Jing and
                  Xiaolei Zhou and
                  Yuxuan Zhang and
                  Yunxia Fan and
                  Jingping Bi},
  title        = {CausalTAD: Causal Implicit Generative Model for Debiased Online Trajectory
                  Anomaly Detection},
  booktitle    = {40th {IEEE} International Conference on Data Engineering, {ICDE} 2024,
                  Utrecht, The Netherlands, May 13-16, 2024},
  pages        = {4477--4490},
  publisher    = {{IEEE}},
  year         = {2024}
}
```
