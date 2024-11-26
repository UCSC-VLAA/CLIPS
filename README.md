# **CLIPS**

**Official implementation of the paper "_CLIPS: An Enhanced CLIP Framework for Learning with Synthetic Captions_".**

![Method Pipeline](./docs/resources/method.jpg)

Previous works show that noisy, web-crawled image-text pairs may limit vision-language pretraining like CLIP and propose learning with synthetic captions as a promising alternative. Our work continues this effort, introducing two simple yet effective designs to better leverage richly described synthetic captions:

1. By observing a strong inverse effect with synthetic captions, we use only **partial synthetic captions** to feed the text encoder, achieving significantly better performance.
2. We incorporate an **autoregressive captioner** that mimics the recaptioning process, predicting full-length synthetic captions conditioned on the image and original web-crawled captions.

Our method achieves **state-of-the-art (SOTA)** results in zero-shot image-text retrieval on MSCOCO and Flickr30K, while enhancing the visual capability of LLaVA.

---

## **Links**
- [📄 Paper (arXiv)](https://arxiv.org/abs/2406.08478)  
- [🤗 Pretrained Model on HuggingFace](https://huggingface.co/UCSC-VLAA/ViT-L-14-CLIPS-Recap-DataComp-1B)  
- [🌐 Project Page](https://ucsc-vlaa.github.io/CLIPS/)

---

## **Proposed Method**

---

## **Key Results**

### **Inverse Effect with Synthetic Captions**
![Inverse Effect Visualization](./docs/resources/mask_strategy.jpg)

Visualization of four different token reduction strategies. These strategies can improve the model's learning efficiency on synthetic captions to varying degrees. Among these strategies, the sub-caption and block mask perform best.

---

### **Zero-Shot Cross-Modal Retrieval**
![Zero-Shot Retrieval Results](./docs/resources/retrieval.png)

Our method consistently achieves superior performance across all benchmarks and model sizes, yielding significant improvements over the baselines.

---

### **Comparison with State-of-the-Art Methods**
![SOTA Comparison](./docs/resources/sota.png)

With increased computational resources and scaling, our best model further achieves 76.4% and 96.6% R@1 text retrieval performance on MSCOCO and Flickr30K respectively, and 57.2% and 83.9% R@1 image retrieval performance on the same datasets, setting new state-of-the-art (SOTA) results.

---

### **CLIPS in LLaVA**
![LLaVA Results](./docs/resources/LLaVA.png)

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
