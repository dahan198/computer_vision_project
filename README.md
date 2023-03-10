# Online Action Segmentation Transformer

## 1. Introduction
Action segmentation in untrimmed videos became very important in many fields, such as surgery, where trainee surgeons learn by observing experts.
Algorithms for action segmentation tasks are typically used in offline mode, and those which are offline does not use Attention and Transformers.
We explored the Action Segmentation Transformer (ASFormer) and turn it into an online model.

## 2. Dataset
The dataset contains 96 videos of different lengths. In every video, one person performs a simulated procedure which is composed of 6 basic gestures: Needle passing, Pull the suture, Instrumental tie, Lay the knot, Cut the suture, and a gesture which is called ‘no gesture’ and does not represent any specific motion.

<p align="center">
  ![Snapshots from the dataset](https://user-images.githubusercontent.com/30556126/224326369-a57a9865-c539-4418-823c-2f0e0c1f0ba1.png)
</p>

## 3. Architecture
### 3.1. Architecture Description
ASFormer is a transformer-based architecture designed specifically for action segmentation tasks.
Lack of inductive biases - the action segmentation task is known for its small-size datasets, which causes the lack of inductive bias. To address this issue, additional temporal convolutions were applied (Dilated Conv, Figure 4) in each layer used as inductive priors, utilizing the temporal structure of the problem.
Difficulty in forming effective representations - due to the long input sequence. Hierarchical self-attention and cross-attention layers were used to address this issue. This representation captures global and local representations, enables high convergence, and reduces total space and time complexity.

