# **CLIPS**

**Official implementation of the paper "_CLIPS: An Enhanced CLIP Framework for Learning with Synthetic Captions_".**

---

## **Authors**

- [Yanqing Liu](https://yanqing0327.github.io/Yanqing.github.io/)<sup>1</sup>, [Xianhang Li](https://xhl-video.github.io/xianhangli/)<sup>1</sup>, [Zeyu Wang](https://zw615.github.io/)<sup>1</sup>,  [Bingchen Zhao](https://bzhao.me/)<sup>2</sup>, [Cihang Xie](https://cihangxie.github.io/)<sup>1</sup>  

<sup>1</sup>UC Santa Cruz, <sup>2</sup>University of Edinburgh  

---

## **Links**
- [📄 Paper (arXiv)](https://arxiv.org/abs/2406.08478)  
- [🤗 Pretrained Model on HuggingFace](https://huggingface.co/UCSC-VLAA/ViT-L-14-CLIPS-Recap-DataComp-1B)  

---

## **Proposed Method**

### **CLIPS Pipeline**
<img src="./docs/resources/method.jpg" alt="Method Pipeline" style="width: 40%; display: block; margin: 0 auto;" />

Previous works show that noisy, web-crawled image-text pairs may limit vision-language pretraining like CLIP and propose learning with synthetic captions as a promising alternative. Our work continues this effort, introducing two simple yet effective designs to better leverage richly described synthetic captions:

1. By observing a strong inverse effect with synthetic captions, we use only **partial synthetic captions** to feed the text encoder, achieving significantly better performance.
2. We incorporate an **autoregressive captioner** that mimics the recaptioning process, predicting full-length synthetic captions conditioned on the image and original web-crawled captions.

Our method achieves **state-of-the-art (SOTA)** results in zero-shot image-text retrieval on MSCOCO and Flickr30K, while enhancing the visual capability of LLaVA.

---

## **Key Results**

### **Inverse Effect with Synthetic Captions**
<img src="./docs/resources/mask_strategy.jpg" alt="Inverse Effect Visualization" style="width: 50%; display: block; margin: 0 auto;" />

Visualization of four different token reduction strategies. These strategies can improve the model's learning efficiency on synthetic captions to varying degrees. Among these strategies, the sub-caption and block mask perform best.

---

### **Zero-Shot Cross-Modal Retrieval**
<img src="./docs/resources/retrieval.png" alt="Zero-Shot Retrieval Results" style="width: 50%; display: block; margin: 0 auto;" />

Our method consistently achieves superior performance across all benchmarks and model sizes, yielding significant improvements over the baselines.

---

### **Comparison with State-of-the-Art Methods**
<img src="./docs/resources/sota.png" alt="SOTA Comparison" style="width: 50%; display: block; margin: 0 auto;" />

With increased computational resources and scaling, our best model further achieves 76.4% and 96.6% R@1 text retrieval performance on MSCOCO and Flickr30K respectively, and 57.2% and 83.9% R@1 image retrieval performance on the same datasets, setting new state-of-the-art (SOTA) results.

---

### **CLIPS in LLaVA**
<img src="./docs/resources/LLaVA.png" alt="LLaVA Results" style="width: 50%; display: block; margin: 0 auto;" />

Replacing OpenAI-CLIP with **CLIPS** significantly boosts LLaVA's performance across various benchmarks.

---

## **Model Zoo**

| Model          | Link                                                                                     |
|----------------|------------------------------------------------------------------------------------------|
| CLIPS-Large-14 | [🤗 HuggingFace Model](https://huggingface.co/UCSC-VLAA/ViT-L-14-CLIPS-Recap-DataComp-1B) |
| CLIPS-Huge-14  | Coming Soon...                                                                          |

<!-- ---

## **Citation**

If you use our work, please cite it:

```bibtex
@article{liu2024clips,
  title={CLIPS: An Enhanced CLIP Framework for Learning with Synthetic Captions},
  author={Liu, Yanqing and Li, Xianhang and Wang, Zeyu and Zhao, Bingchen and Xie, Cihang},
  journal={arXiv preprint arXiv:2406.08478},
  year={2024}
} -->
