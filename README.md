# Online Action Segmentation Transformer

## 1. Introduction
Action segmentation in untrimmed videos became very important in many fields, such as surgery, where trainee surgeons learn by observing experts.
Algorithms for action segmentation tasks are typically used in offline mode, and those which are offline does not use Attention and Transformers.
We explored the Action Segmentation Transformer (ASFormer) and turn it into an online model.

## 2. Dataset
The dataset contains 96 videos of different lengths. In every video, one person performs a simulated procedure which is composed of 6 basic gestures: Needle passing, Pull the suture, Instrumental tie, Lay the knot, Cut the suture, and a gesture which is called ‘no gesture’ and does not represent any specific motion.


<p align="center" width="100%">
    <img width="50%" src="https://user-images.githubusercontent.com/30556126/224326369-a57a9865-c539-4418-823c-2f0e0c1f0ba1.png">
</p>

<p align="center" width="100%">
    <img width="50%" src="https://user-images.githubusercontent.com/30556126/224329116-f08bbe8e-38dc-433c-82e9-fd368e8c6643.png">
</p>

## 3. Architecture
### 3.1. Architecture Description
ASFormer is a transformer-based architecture designed specifically for action segmentation tasks. While transformers have had significant breakthroughs in natural language processing, they have been less commonly used for video tasks, especially with action segmentation tasks. The challenges in action segmentation tasks that ASFormer aims to address are:

 Markup : * Bullet list
              * Nested bullet
                  * Sub-nested bullet etc
          * Bullet list item 2

Usually, transformer models requires a lot of data in order to achieve good results, and the action segmentation task is known for its small-size datasets, which causes the lack of inductive bias. To address this issue, additional temporal convolutions were applied (Dilated Conv) in each layer used as inductive priors, utilizing the temporal structure of the problem.
Difficulty in forming effective representations - due to the long input sequence. Hierarchical self-attention and cross-attention layers were used to address this issue. This representation captures global and local representations, enables high convergence, and reduces total space and time complexity.

