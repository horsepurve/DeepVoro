# DeepVoro: Few-shot Learning as Cluster-induced Voronoi Diagrams (ICLR 2022)

<div align="center">
  <a href="https://openreview.net/forum?id=6kCiVaoQdx9">OpenReview</a>
</div>
<div align="center">
  <a href="https://arxiv.org/abs/2202.02471">arXiv</a>
</div>

## Introduction

In metric-based Few-Shot Learning (FSL), the classification model (e.g. Prototypical Networks and Logistic Regression) can be intepreted as either a [Voronoi diagram (VD)](https://en.wikipedia.org/wiki/Voronoi_diagram), or a [Power diagram (PD)](https://en.wikipedia.org/wiki/Power_diagram) constructed in the feature space.

In this work we dive deeper, geometrically, into this line and extend VD/PD to Cluster-induced Voronoi Diagrams (CIVD), proposed in [FOCS 2013](https://ieeexplore.ieee.org/document/6686175) and [SIAM Journal on Computing 2017](https://epubs.siam.org/doi/pdf/10.1137/15M1044874) ([PDF](https://ieeexplore.ieee.org/iel7/6685222/6686124/06686175.pdf?casa_token=GGuzxr8aLFIAAAAA:Rd0PS1RlLftuYLlDvmaKV9Y-FhKv9cZPmvADugH5YdREm5KgTWwTcDdVYqujrxI06-Pxi4RmCA)), and establish DeepVoro to combat the extreme data insufficiency in FSL.

<p align="center">
  <img src="./img/demo_mnist.png">
</p>

## Prerequisites

Install dependencies via

```bash
pip install -r requirements.txt
```

## Reproducing the results

In this tutorial we use CUB dataset, the smallest among three, to illustrate the DeepVoro workflow.

## References

If you find the software useful please consider citing:

```
@misc{ma2022fewshot,
      title={Few-shot Learning as Cluster-induced Voronoi Diagrams: A Geometric Approach},
      author={Chunwei Ma and Ziyun Huang and Mingchen Gao and Jinhui Xu},
      year={2022},
      eprint={2202.02471},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

or

```
@inproceedings{ma2022fewshot,
    title={Few-shot Learning via Dirichlet Tessellation Ensemble},
    author={Chunwei Ma and Ziyun Huang and Mingchen Gao and Jinhui Xu},
    booktitle={International Conference on Learning Representations},
    year={2022},
    url={https://openreview.net/forum?id=6kCiVaoQdx9}
}
```

**Acknowledgments**

In this project we use (parts of) the official implementations of the following works:

* [S2M2_fewshot](https://github.com/nupurkmr9/S2M2_fewshot)
* [simple_shot](https://github.com/mileyan/simple_shot)
* [Few_Shot_Distribution_Calibration](https://github.com/ShuoYang-1998/Few_Shot_Distribution_Calibration)

We thank the respective authors for open-sourcing their methods, which makes this work possible.

If you have any problem please [contact me](mailto:chunweim@buffalo.edu).
