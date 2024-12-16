# CausalTAD
Code for ICDE'24 paper: "CausalTAD: Causal Implicit Generative Model for Debiased Online Trajectory Anomaly Detection" and several other trajectory anomaly detection models: SAE, VSAE, GM-VSAE.

# Acknowledgements
The implementation of GM-VSAE is based on the code from [this repository](https://github.com/chwang0721/GM-VSAE), I simplified the code for pretraining and training Gaussian Mixture Model to make the logic of model optimization more straightforward, and I modified the code of model inference based on my own understanding.

# Citation
Please cite our paper if our paper or codes is helpful.

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
