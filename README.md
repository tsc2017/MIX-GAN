# MIX-GAN
Some recent state-of-the-art generative models in ONE notebook.

This repo implements any method that can match the following regular expression:

`(MIX-)?(GAN|WGAN|BigGAN|MHingeGAN|AMGAN|StyleGAN|StyleGAN2)(\+ADA|\+CR|\+EMA|\+GP|\+R1|\+SA|\+SN)*`

code for the paper [**Lessons Learned from the Training of GANs on Artificial Datasets**](https://arxiv.org/abs/2007.06418) and beyond
# Free GPU training on Colab
[![Example In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tsc2017/MIX-GAN/blob/main/MIX-MHingeGAN-CIFAR-10.ipynb)

This implemetation supports automatic mixed-precision training of TensorFlow, which can reduce GPU memory usage and training time dramatically. Therefore, it is recommended to upgrade to [Colab Pro](https://colab.research.google.com/signup) in order to use GPUs with Tensor Cores.
# Free TPU training on Colab
Coming soon...
# Training on Cloud TPUs
- First [disable Stackdriver Logging](https://console.cloud.google.com/logs/router?) to avoid unnecessary charges.
- [Create cloud TPUs](https://cloud.google.com/tpu/docs/creating-deleting-tpus), TPU software version should be at least `2.4.0` or `nightly`.
- Fill in `TPU_NAMES` and `ZONE` in the  the above notebook for TPUs. Set up envirionment variables `LOG` and `DATA`, run the notebook.
- [Delete TPUs](https://cloud.google.com/tpu/docs/creating-deleting-tpus).
