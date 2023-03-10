# Online Action Segmentation Transformer

## 1. Introduction
<p align="justify">
Action segmentation in untrimmed videos became very important in many fields, such as surgery, where trainee surgeons learn by observing experts.
Algorithms for action segmentation tasks are typically used in offline mode, and those which are offline does not use Attention and Transformers.
We explored the Action Segmentation Transformer (ASFormer) and turn it into an online model.
</p>

## 2. Dataset
<p align="justify">
The dataset contains 96 videos of different lengths. In every video, one person performs a simulated procedure which is composed of 6 basic gestures: Needle passing, Pull the suture, Instrumental tie, Lay the knot, Cut the suture, and a gesture which is called ‘no gesture’ and does not represent any specific motion.
</p>

<p align="center" width="100%">
    <img width="50%" src="https://user-images.githubusercontent.com/30556126/224326369-a57a9865-c539-4418-823c-2f0e0c1f0ba1.png">
</p>

<p align="center" width="100%">
    <img width="50%" src="https://user-images.githubusercontent.com/30556126/224329116-f08bbe8e-38dc-433c-82e9-fd368e8c6643.png">
</p>

## 3. Architecture
### 3.1. Architecture Description
<p align="justify"> 
ASFormer is a transformer-based architecture designed specifically for action segmentation tasks. While transformers have had significant breakthroughs in natural language processing, they have been less commonly used for video tasks, especially with action segmentation tasks. The challenges in action segmentation tasks that ASFormer aims to address are:

- **Lack of inductive biases** - Usually, transformer models requires a lot of data in order to achieve good results, and the action segmentation task is known for its small-size datasets, which causes the lack of inductive bias. To address this issue, additional temporal convolutions were applied (Dilated Conv) in each layer used as inductive priors, utilizing the temporal structure of the problem.
- **Difficulty in forming effective representations** - Due to the long input sequence. Hierarchical self-attention and cross-attention layers were used to address this issue. This representation captures global and local representations, enables high convergence, and reduces total space and time complexity.
- **Does not meet the refinement demand of the action segmentation task** - To address this issue, additional decoders were added. Each decoder uses a cross-attention mechanism to bring in information from the last encoder/decoder.
</p>

<p align="center" width="100%">
    <img width="80%" src="https://user-images.githubusercontent.com/30556126/224336960-c47f50ff-6883-40ba-bac5-7813958d86bb.png">
</p>

## 3.2. Our Adaptation
<p align="justify">
In the original ASFormer, information sharing between frames only occurs through hierarchical self/cross attention. To achieve the online property, and ensure that only future data within a specific window is considered, we zero out all neighboring frames found after the future window.
</p>

<p align="center" width="100%">
    <img width="60%" src="https://user-images.githubusercontent.com/30556126/224334975-843c88aa-06e2-4937-a9d9-d4463b5f6c02.png">
</p>
<p align="justify">
The figure illustrates our adaptation for a future window of size two. Each node in the figure represents a frame, and the number on the node indicates the frame's location in the video length. Each row of nodes represents a level in the hierarchy. The pink nodes in the figure were zeroed out to ensure that only the relevant frames in the future window are considered. Upper: illustrates our adaptation for the frame in the 0-location. Lower: illustrates our adaptation for the frame in the 20-location.
</p>

## 4. Metrics
<p align="justify">
For the segmentation metric, we used the segmental overlaps F1@k where k ∈ {10, 25, 50} and the segmental edit distance score.
For the frame-wise metric, accuracy was used.
</p>

## 5. Results
<p align="justify">
We observed consistent results between the training, validation, and test for the frame-wise metric (accuracy) and the loss, which mainly contained a classification error. The accuracy and loss were improved as the window size increased.
However, the segmentation metric values were less consistent with the window size: The edit distance showed better results for future windows in sizes 7, 15, and 30 than “all future”. Similarly, the F1 metrics also presented the same phenomena. One reason for that may be due to the loss function and the smooth factor that prioritizes the classification error over smoothness and focuses on frame-wise performance. We can see that for the largest sizes, we got much better results than the smallest window sizes, such as 0 and 3, so there is an effect on a small range of window sizes.
We observed that the model converged quickly during training, maybe due to the ASFormer architecture's power or the problem's relatively low complexity. These results present that our online algorithm with limited future window sizes can achieve comparable performance to the offline approach with all future information in this problem.
</p>

<p align="center" width="100%">
    <img width="60%" src="https://user-images.githubusercontent.com/30556126/224337798-3cfb6c5c-fba5-468b-a866-d50126523cc1.png">
    <img width="80%" src="https://user-images.githubusercontent.com/30556126/224337814-5ef6e85b-71f4-487e-bb19-e4d5e919c53a.png">
    <img width="50%" src="https://user-images.githubusercontent.com/30556126/224337827-51d56212-3eee-418f-8058-9f7abf2c5e51.png">
</p>


## 6. Trade-Off
<p align="justify">
We conducted seven experiments with varying future window sizes: 0s, 0.2s, 0.47s, 0.73s, 1s, 2s, and all-future. We observed a trade-off between the algorithm's performance and the future window size. As the future window size increased, the algorithm's performance improved for the frame-wise metric. However, for segmentation metrics, we found that there is a trade-off for smaller future window sizes. For larger future window sizes, the algorithm's performance did not improve and even worsened.
</p>

<p align="center" width="100%">
<img width="60%" src="https://user-images.githubusercontent.com/30556126/224338044-66fdc760-883a-4c4e-b2ac-382a899ca603.png">
</p>

## 7. Discussion
<p align="justify"> 
We conducted a series of experiments to evaluate the performance of our online algorithm with varying future window sizes. We also compared our online algorithm to an offline mode. The results of our experiments showed that the accuracy and loss were improved as the future window size increased, while the segmentation metrics were less consistent with the window size.
Our online algorithm with limited future window sizes can achieve comparable performance to the offline approach in this particular problem. Furthermore, our project demonstrates how the ASFormer architecture can be adapted to online mode. It can be very useful for other transformer architectures and other attention-based mechanisms from many fields, and thus it can expand beyond computer vision tasks.
</p>


