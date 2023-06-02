# TranS4mer
This is the PyTorch Implementation of **Efficient Movie Scene Detection using State-Space Transformers (TranS4mer)** [[arxiv]](https://arxiv.org/abs/2212.14427)

## 1. Data and Environmental Setup
We have tested the implementation on the following environment:
  * Python 3.8.12 / PyTorch 1.10.0 / torchvision 0.11.1 / CUDA 11.3

The code is based on [BaSSL](https://github.com/kakaobrain/bassl) 
Follow [BaSSL](https://github.com/kakaobrain/bassl) for environmental setup and data download.

Also follow [S4](https://github.com/HazyResearch/state-spaces) for installation regarding S4 models.

## 2. Training

**(1) Pre-training BaSSL**  
`cd trans4mer; bash ../scripts/run_pretrain_bassl.sh`  

**(2) Finetuning and Evaluation**

`cd trans4mer; bash ../scripts/run_finetune.sh`

## 3. Pre-trained Models

TODO

## 4. Citation
If you find this code helpful for your research, please cite our paper.
```
@inproceedings{islam2023efficient,
  title={Efficient Movie Scene Detection using State-Space Transformers},
  author={Islam, Md Mohaiminul and Hasan, Mahmudul and Athrey, Kishan Shamsundar and Braskich, Tony and Bertasius, Gedas},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={18749--18758},
  year={2023}
}
```
