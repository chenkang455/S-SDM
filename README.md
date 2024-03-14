# S-SDM

This is the official PyTorch implementation of [SpikeReveal: Unlocking Temporal Sequences from Real Blurry Inputs with Spike Streams]().

> Reconstructing a sequence of sharp images from the blurry input is crucial for enhancing our insights into the captured scene and poses a significant challenge due to the limited temporal features embedded in the image. Spike cameras, sampling at rates up to 40,000 Hz, have proven effective in capturing motion features and beneficial for solving this ill-posed problem. Nonetheless, existing methods fall into the supervised learning paradigm, which suffers from notable performance degradation when applied to real-world scenarios that diverge from the synthetic training data domain. Moreover, the quality of reconstructed images is capped by the generated images based on motion analysis interpolation, which inherently differs from the actual scene, affecting the generalization ability of these methods in real high-speed scenarios.  To address these challenges, we propose the first self-supervised framework for the task of spike-guided motion deblurring.  

Sequence reconstruction comparison, where 1 blurry frame corresponds to 300 sharp latent frames. (flicker is caused by the gif compression)

<img src="imgs/middle_calib_compress.gif" style="width: 100%;">

- [x] Release the scripts for simulating GOPRO dataset.
- [ ] Release the synthetic/real-world dataset.
- [ ] Release the training and testing code.
- [ ] Release the pretrained model.

## Dataset

Guidance on synthesizing the spike-based GOPRO dataset can be found in [GOPRO_dataset](scripts/GOPRO_dataset.md). 

Converted GOPRO dataset and real-world blur RSB dataset will be available soon.

## Installation

todo

## Training

todo

## Evaluation

todo

## Contact
Should you have any questions, please feel free to contact [mrchenkang@whu.edu.cn](mailto:mrchenkang@whu.edu.cn).

