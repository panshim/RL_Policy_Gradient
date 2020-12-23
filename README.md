# RL_Policy_Gradient
- [RL_Policy_Gradient](#rl_policy_gradient)
  - [1. How to set up the environment?](#1-how-to-set-up-the-environment)
  - [2. Overview](#2-overview)
    - [2.1 Environment](#21-environment)
    - [2.2 Problem Formulation](#22-problem-formulation)
  - [3. Our Extensions](#3-our-extensions)
    - [3.1 DAPG verification](#31-dapg-verification)
    - [3.2 Off-Policy Sampling](#32-off-policy-sampling)
    - [3.3 Bootstrapping](#33-bootstrapping)
    - [3.4 Recurrent Plolicy](#34-recurrent-plolicy)
    - [4. Conclusion](#4-conclusion)

## 1. How to set up the environment?
* This project is an extension based on [Aravind Rajeswaran's project](https://github.com/aravindr93/hand_dapg). For the detailed setup instructions and dependencies, please refer to his project.
* This repo only includes the files created/modified by us. So if you want to run our code, please first download the code by Aravind, and substitute the original code with ours.

## 2. Overview
### 2.1 Environment
![task](presentation_report/task.png)
*Fig. 1. In-hand manipulation task, whose goal is to re-position the blue pen to match the orientation of the green target. (Only the orientation, the target position is defined elsewhere)*

To setup our experiments, we use an external customized [OpenAI Gym environment](https://github.com/vikashplus/mj_envs) simulated with [MuJoCo physics engine](http://www.mujoco.org)

The platform simulates the analogue of a 24 DoF ADROIT robot hand as the controlled agent for the task. As shown in Fig.2, the blue arrows mark the 24 controllable hand joints. Besides, the platform also provides very realistic simulations of physics by carefully modeling the kinematics, dynamics and sensing details. It enables us to access the measurements of joint positions, object poses, etc, and to visualize the demonstrations of any given policies.

### 2.2 Problem Formulation
![hand](presentation_report/hand.jpg)\
*Fig. 2. Simulated ARDOIT robot hand. The blue arrows represent each of the controllable hand joints.*
The problem is formulated as an episodic task. The episode will terminate either when a failure occurs or it exceeds the preset maximum steps (e.g. 200). Specifically, the failure state is defined that the pen falls from the hand, ie. the height of the pen position is less than a threshold.



The associated Markov Decision Process (MDP) is defined as ![47](teximg/tex_img_2_NVTRW.png) is the state space, ![0](teximg/tex_img_0_P4HBH.png) is the action space, ![1](teximg/tex_img_1_WFF9W.png) is the reward, ![1](teximg/tex_img_6_WSAP0.png) is the transition dynamics unknown to us, and ![48](teximg/tex_img_7_06EPY.png) is the discount factor.

The goal of this problem is to find the optimal stochastic policy such that it maximizes the expected discounted returns:

![48](teximg/tex_img_3_BJBQO.png)

where ![48](teximg/tex_img_9_VCFV8.png) is the reward received at step ![2](teximg/tex_img_10_8WMZQ.png), and ![3](teximg/tex_img_11_TYH06.png) is the length of the episode.

For our specific application, the detailed definitions for the components in our problem formulation are described below:


* **State**: The state is continuous represented by a 45-dimensional vector (![4](teximg/tex_img_13_FEJYR.png)). Each of the dimensions represents one of the measurements including hand joint angles (24 DoF), pen pose (6 DoF), pen linear velocity (3 DoF), target pen orientation (3 DoF), and difference of current and target pen pose (6 DoF). These measurements can be directly observed and accessed from the simulator.
* **Action**: The action is also continuous, represented by a 24-dimensional vector (![5](teximg/tex_img_20_38868.png)). Each of the dimensions is defined as the desired angle at the next step for one of the 24 joints on the robot hand.
* **Reward**: The reward we use is heavily shaped to help with faster learning. The shaping of the reward is adopted from the default setting of the environment we use. Basically, the reward is a function of the state, mainly related to the difference of current and target pen pose. Additionally, bonus for success and penalty for failure are both applied in shaping the reward. The reward can be explicitly expressed as

![6](teximg/tex_img_22_7PKZY.png)

![7](teximg/tex_img_23_1D9ZB.png)

where ![8](teximg/tex_img_24_OC076.png), ![9](teximg/tex_img_25_JJYVN.png), ![10](teximg/tex_img_26_XYRYB.png), ![11](teximg/tex_img_27_AW7UA.png) respectively denote the current position, target position, current orientation and target orientation of the pen at time ![2](teximg/tex_img_10_8WMZQ.png); ![12](teximg/tex_img_29_LPIM3.png) is the height of the pen at time ![2](teximg/tex_img_10_8WMZQ.png); and ![13](teximg/tex_img_31_QZ72D.png) denotes the inner product of two given vectors.


## 3. Our Extensions
### 3.1 DAPG verification
![verify](presentation_report/verify.png)
*Fig. 3. Evaluations in experiments verifying effectiveness of different strategies used in DAPG.*

The DAPG algorithm is derived from the classical REINFORCE with Baseline, which is a Monte-Carlo policy gradient method for optimizing the stochastic policy. It involves one neural network for the value-function baseline ![14](teximg/tex_img_32_0M3QR.png) and another network for the policy ![15](teximg/tex_img_33_PQUM8.png). At the beginning of each training iteration, it will first sample a batch of trajectories using current policy, and estimate the advantage ![16](teximg/tex_img_34_UT11N.png) of each step with the actual return ![17](teximg/tex_img_35_DTB40.png) by

![18](teximg/tex_img_36_11PG0.png)

And then the baseline network weights ![19](teximg/tex_img_37_8CTNZ.png) and the policy network weights ![20](teximg/tex_img_38_HE861.png) will be successively updated with a small step size ![21](teximg/tex_img_39_SEL9B.png) and ![22](teximg/tex_img_40_GT9J2.png) by

![23](teximg/tex_img_41_KYZ6K.png)

For the major contributions of DAPG, it adopts additional training strategies which are crucial for fast convergence. These strategies can be summarized as the followings:
\begin{enumerate}
* Pre-training with behavior cloning. Utilizing an expert demonstration dataset ![24](teximg/tex_img_42_HVRW3.png) which includes 25 pre-collected trajectories, the policy network will be pre-trained with supervised learning by maximizing the likelihood of actions observed in the demonstration dataset:
![25](teximg/tex_img_44_F2U2H.png)

* Natural policy gradient (NPG). In the fine-tuning phase of the training, instead of using the vanilla REINFORCE gradient ![26](teximg/tex_img_45_A02IE.png) expressed in Eq. 7, it will make a normalized gradient ascent update to the policy pre-conditioned with the Fisher Information Matrix ![27](teximg/tex_img_46_P8K8R.png):
![28](teximg/tex_img_47_7K96L.png)

* Augmented policy gradient from demo data. In fine-tuning, besides using the newly sampled data at each training iteration to compute the policy gradient, DAPG also keeps utilizing the demonstration data to help shape the gradient. Briefly speaking, this is done by merging the gradient computed from the demonstration data into the original gradient computed from the newly sampled data with a small decaying weight. This strategy can elegantly guide the exploration at the early stage of fine-tuning. *This technique is able to better shape the policy gradient and make faster progress on the policy improvement.*

The results of our verifying experiments are shown in Fig.3, where the plotted curves show how the performance evolves with respect to training iterations. For better visualization, all the curves are smoothed by averaging over successive 5 iterations. We use two evaluations to compare the performance of different methods. **One metric is the success rate**. It is considered as a success when the difference of the pen pose and its target is within a small tolerance at the end of the episode. **Another metric is the score**, which is the averaged total rewards among the evaluation episodes.

From the experiment results, compared with learning from scratch, behavior cloning with demonstrations is able to provide a good initialization of the policy network. And compared with the vanilla REINFORCE which improves very slow, NPG is pretty effective on helping with fast convergence.  


### 3.2 Off-Policy Sampling
![off_policy](presentation_report/off_policy.png)
*Fig. 4. Evaluations in experiments on off-policy sampling*

The first potential problem of the original DAPG is that it uses on-policy reinforcement learning for fine-tuning. This could be data inefficient, and more importantly, it might not be able to sufficiently explore the region near the optimal policy when the demonstration data used for pre-training is not optimal.

Therefore, we propose to encourage more off-policy exploration by sampling with a mixture policy ![29](teximg/tex_img_49_XYZ38.png), which combines the current policy ![30](teximg/tex_img_50_8I25B.png) and the policy from the previous iteration ![31](teximg/tex_img_51_FKNXC.png) with a weight ![32](teximg/tex_img_52_KQJWK.png). The results are shown in Fig.~\ref{fig:off_policy}. It can be observed that this simple off-policy sampling improves the performance a little in the early iterations, but it does not make a big difference afterwards. 

### 3.3 Bootstrapping
![bootstrapping](presentation_report/bootstrapping.png)
*Fig. 5. Evaluations in experiments on bootstrapping.*

In policy gradient methods, the estimation of the advantage function is important to policy improvement since it is highly related to estimating the policy gradient. As introduced previously, DAPG estimates the advantage by Monte-Carlo, which is unbiased due to the use of actual returns. However, this estimation is presumed to be of high variance, and is thus likely to slow down the training.

To deal with this probable high variance of advantage estimation in DAPG, we propose to use Temporal Difference (TD) bootstrapping. One simple example of bootstrapping is Actor-Critic, which uses one-step TD to estimate the advantage:
![33](teximg/tex_img_53_UMQH5.png)

This estimation in Actor-Critic is able to achieve low variance, but it is also presumably bad as verified in Fig.5, since the bias is high due to the inaccurate approximation of the value-function baseline ![34](teximg/tex_img_54_SURCG.png). However, if the estimation of the advantage is well balanced between bias and variance, the training process will potentially benefit a lot from it. Therefore, we finally apply Generalized Advantage Estimation (GAE), which is a useful technique to adjust the bias-variance trade-off.

GAE is defined as the exponentially-weighted average of k-step TD estimators, which can be simplified as:

![35](teximg/tex_img_55_1IFXT.png)

where ![36](teximg/tex_img_56_I75L7.png) is the k-step TD residual:

![37](teximg/tex_img_57_7EW33.png)

To note, ![38](teximg/tex_img_58_HMVI7.png) is exactly the one-step TD same as in Actor-Critic; in contrast, ![39](teximg/tex_img_59_2EU54.png) is ![40](teximg/tex_img_60_66QI2.png)-just, and thus has low bias regardless of the accuracy of ![41](teximg/tex_img_61_KO08K.png), but it has high variance due to the sum of terms. With carefully tuned ![42](teximg/tex_img_62_T0H5O.png), we can successfully improve the performance of DAPG by a great margin, as shown in Fig.5

### 3.4 Recurrent Plolicy
The policy used in DAPG only depends on the current state and has no memory of previous states. This assumption can become too simple to make good decisions in practice. Intuitively, a better decision could be made if some important information obtained through interactions with the environment in previous steps, such as the weight and implicit shape of the object, can propagate forwards.

Therefore, we propose to replace the original policy network modeled by a 3-layer fully-connected neural network with Long Short-Term Memory (LSTM). architecture as shown in Fig.6
![policy_net](presentation_report/policy_net.png)
*Fig. 6. Replacement of the policy network with LSTM architecture. The left is the original fully-connected policy network where the activations are tanh; the right is our newly applied LSTM policy network where the activations are changed to ReLU.*

The experimental results are shown in Fig.7, it can be obviously observed that the newly applied LSTM policy eventually helps accelerate the learning.
![lstm](presentation_report/lstm.png)
*Fig. 7. Evaluations in experiments on LSTM policy*

### 4. Conclusion
* (Behavior cloning)Pre-training with demonstrations is very helpful for initializing the policy
* GAE and LSTM policy both help accelerate the learning, and outperform the original DAPG settings in terms of the success rate and total rewards. (Shown in Fig.8)
* Mixed off-policy sampling's performance is almost the same with the original on-policy DAPG algorithm
* For the bootstrapping extension, We did several experiments to search over different values of ![43](teximg/tex_img_64_NGP7Z.png) with fixed ![44](teximg/tex_img_65_2QXJ6.png). Particularly, we tested with ![45](teximg/tex_img_66_CC1MQ.png)= 0 (Actor-Critic), 0.1, 0.3, 0.5, 0.8, 0.97, and 1.0. As shown in Fig.9, the best balance between bias and variance is the case of ![46](teximg/tex_img_67_BAY66.png) = 0.97.
* Applying LSTM policy significantly improved the converging speed

For more detailed analysis, please refer to our [report](presentation_report/report.pdf).
![summary](presentation_report/summary.png)
*Fig. 8. Summarized results of experiments on our extensions*

![gae](presentation_report/gae.png)
*Fig. 9. Performance of GAE applied with different λ value.*