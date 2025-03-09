# **Multi-dimensional Visual Prompt Enhanced Image Restoration via Mamba-Transformer Aggregation**

Aiwen Jiang, Hourong Chen, Zhiwei Chen, Jihua Ye, Mingwen Wang，“Multi-dimensional Visual Prompt Enhanced Image Restoration via Mamba-Transformer Aggregation”，arXiv, 2024

### [Multi-dimensional Visual Prompt Enhanced Image Restoration via Mamba-Transformer Aggregation](https://arxiv.org/abs/2412.15845).


 **Abstract**—Image restoration is an important research topic that has wide industrial applications in practice. Traditional deep learning-based methods were tailored to specific degradation type, 
 which limited their generalization capability. Recent efforts have focused on developing ”all-in-one” models that can handle different degradation types and levels within single model. 
 However, most of mainstream Transformer-based ones confronted with dilemma between model capabilities and computation burdens, since self-attention mechanism quadratically increase in computational complexity with respect to image size, and has inadequacies in capturing long-range dependencies. 
 Most of Mamba-related ones solely scanned feature map in spatial dimension for global modeling, failing to fully utilize information in channel dimension. 
 To address aforementioned problems, this paper has proposed to fully utilize complementary advantages from Mamba and Transformer without sacrificing computation efficiency. Specifically, the selective scanning mechanism of Mamba is employed to focus on spatial modeling, enabling capture long-range spatial dependencies under linear complexity.
 The self-attention mechanism of Transformer is applied to focus on channel modeling, avoiding high computation burdens that are in quadratic growth with image’s spatial dimensions.
 Moreover, to enrich informative prompts for effective image restoration, multi-dimensional prompt learning modules are proposed to learn prompt-flows from multi-scale encoder/decoder layers, benefiting for revealing underlying characteristic of various degradations from both spatial and channel perspectives,
 therefore, enhancing the capabilities of ”all-in-one” model to solve various restoration tasks. Extensive experiment results on several image restoration benchmark tasks such as image denoising, dehazing, and deraining, have demonstrated that the proposed method can achieve new state-of-the-art performance, compared with many popular mainstream methods.
 Related source codes and pre-trained parameters will be public on github https://github.com/12138-chr/MTAIR.
 
 Index Terms—Image restoration, All-in-one, Mamba, Transformer, Prompt learning, Low-level vision

 ![image](https://github.com/user-attachments/assets/dd3b152c-a44f-4cc7-9da5-3db9e3b782fd)
 
## Installation

The project is built with Python 3.8, PyTorch 1.8.1. CUDA 10.2, cuDNN 7.6.5
```
pip install -r requirements.txt
```
To use the selective scan(Mamba SSM), the library is needed to install with the folllowing command.
```
pip install causal_conv1d==1.0.0
pip install mamba_ssm==1.0.1
```

## Results

![image](https://github.com/user-attachments/assets/a124abde-d2bb-4d98-a13a-43865d5edb51)
![image](https://github.com/user-attachments/assets/c9201cdb-75b0-4568-a332-bfbef54406c2)
![image](https://github.com/user-attachments/assets/570fb6f7-576e-4124-819f-12028d5d7515)
![image](https://github.com/user-attachments/assets/4c4b3070-bdcd-40b0-9ad9-18a4440c95a2)

## Data Download and Preparation

Denoising: [BSD400](https://drive.google.com/drive/folders/1O1Z8yEbLzndLzk9jK233r8DEI-3Xmeoe?usp=drive_link), [WED](https://drive.google.com/drive/folders/1p7ax2daKZOjHyMA7UFZ3lcoRBeWtTmxn?usp=drive_link), [Urban100](https://drive.google.com/drive/folders/1QgXBf3LOKwZnnWQQBqDt56T630mq_o7v?usp=drive_link), [CBSD68](https://drive.google.com/drive/folders/1mgEDilXcRkE6bkQoGkK-wrf-OhkC2CpI?usp=drive_link)

Deraining: [Train100L&Rain100L](https://drive.google.com/drive/folders/1RjrjuGBK0jsQ5a5j1k-clsdxZkrqPQE2?usp=drive_link)

Dehazing: [RESIDE](https://sites.google.com/view/reside-dehaze-datasets/reside-v0) (OTS)

The pre-trained model will be placed in the ckpt folder.The training data should be placed in directory where can be Denoise,Derain or Dehaze. After placing the training data the directory structure would be as follows: data/Train/{task_name}task_name
```
└───Train
    ├───Dehaze
    │   ├───original
    │   └───synthetic
    ├───Denoise
    └───Derain
        ├───gt
        └───rainy
```
The testing data should be placed in the directory wherein each task has a seperate directory. The test directory after setup:test
```
├───dehaze
│   ├───input
│   └───target
├───denoise
│   ├───bsd68
│   └───urban100
└───derain
    └───Rain100L
        ├───input
        └───target
```

## Citation
If you find this project useful for your research, please consider citing:
~~~
@article{jiang2024multi,
  title={Multi-dimensional Visual Prompt Enhanced Image Restoration via Mamba-Transformer Aggregation},
  author={Jiang, Aiwen and Chen, Hourong and Chen, Zhiwen and Ye, Jihua and Wang, Mingwen},
  journal={arXiv preprint arXiv:2412.15845},
  year={2024}
}
