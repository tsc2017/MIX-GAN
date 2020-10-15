# MIX-GAN 
## code for the paper [**Lessons Learned from the Training of GANs on Artificial Datasets**](https://arxiv.org/abs/2007.06418) and beyond
Some recent state-of-the-art generative models in ONE notebook.

This repo implements any method that can match the following regular expression:

`(MIX-)?(GAN|WGAN|BigGAN|MHingeGAN|AMGAN|StyleGAN|StyleGAN2)(\+ADA|\+CR|\+EMA|\+GP|\+R1|\+SA|\+SN)*`

# Major dependencies
- For the GPU implementation, `tensorflow>=2` or `tensorflow-gpu==1.14` (some modifications for the calculation of IS and FID will be necessary, see the other repos of mine).
- For the TPU implemetation, `tensorflow>=2.4` or `tf-nightly` will be necessary.
# Free GPU training on Colab
[![Example In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tsc2017/MIX-GAN/blob/main/MIX-MHingeGAN-CIFAR-10.ipynb)

This implemetation supports automatic mixed-precision training of TensorFlow, which can reduce GPU memory usage and training time dramatically. Therefore, it is recommended to upgrade to [Colab Pro](https://colab.research.google.com/signup) in order to use GPUs with Tensor Cores. Training `MIX-MHingeGAN` with 10 generators and 10 discriminators takes only 1.5 days on a single Tesla V100.
# Free TPU training on Colab
Coming soon...
# Training on Cloud TPUs
- First [disable Stackdriver Logging](https://console.cloud.google.com/logs/router?) to avoid unnecessary charges.
- [Create cloud TPUs](https://cloud.google.com/tpu/docs/creating-deleting-tpus), TPU software version should be at least `2.4.0` or `nightly`.
- Fill in `TPU_NAMES` and `ZONE` in the  the above notebook for TPUs. Set up environment variables `LOG` and `DATA`, run the notebook.
- [Delete TPUs](https://cloud.google.com/tpu/docs/creating-deleting-tpus).
