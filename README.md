## SigFormer: Signature Transformers for Deep Hedging


> **Abstract** Deep hedging is a promising direction in quantitative finance, incorporating models and techniques from deep learning research. While giving excellent hedging strategies, models inherently requires careful treatment in designing architectures for neural networks. To mitigate such difficulties, we introduce SigFormer, a novel deep learning model that combines the power of path signatures and transformers to handle sequential data, particularly in cases with irregularities. Path signatures effectively capture complex data patterns, while transformers provide superior sequential attention. Our proposed model is empirically compared to existing methods on synthetic data, showcasing faster learning and enhanced robustness, especially in the presence of irregular underlying price data. Additionally, we validate our model performance through a real-world backtest on hedging the SP 500 index, demonstrating positive outcomes.


![Sigformer](https://github.com/anh-tong/sigformer/raw/main/assets/sigformer.jpg)

### Installation

There are major required package including
```
jax
equinox   # https://github.com/patrick-kidger/equinox
signax    # https://github.com/anh-tong/signax
numpyro   # https://github.com/pyro-ppl/numpyro
```

Install via
```
cd sigformer
pip install -e .
```


### Demo

Data for all experiments can be obtained [here](https://drive.google.com/file/d/1VnXHy1ephw85sP-32E74m6uMjsjbxVRN/view?usp=share_link)

A quick demo of hedging rough Bergomi price model is done by executing

```
python scripts/main.py
```

### Citation

```
@article{sigformer,
  author    = {Anh Tong and Thanh Nguyen-Tang and Dongeun Lee and Toan Tran and Jaesik Choi},
  title     = {SigFormer: Signature Transformers for Deep Hedging},
  journal   = {4th ACM International Conference on AI in Finance},
  pdf       = {https://arxiv.org/pdf/2310.13369.pdf},
  abbr      = {ICAIF},
  year      = {2023}
}
```
