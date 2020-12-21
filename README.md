# RL_Policy_Gradient
## 1. Dependency


## 2. Overview
### 2.1 Environment
![task](presentation_report/task.png)
*Fig. 1. In-hand manipulation task, whose goal is to re-position the blue pen to match the orientation of the green target.*


### 2.2 Problem Formulation
![hand](presentation_report/hand.jpg)\
*Fig. 2. Simulated ARDOIT robot hand. The blue arrows represent each of the controllable hand joints.*



## 3. Our Extensions
### 3.1 DAPG verification
![verify](presentation_report/verify.png)
*Fig. 3. Evaluations in experiments verifying effectiveness of different strategies used in DAPG.*

### 3.2 Off-Policy Sampling
![off_policy](presentation_report/off_policy.png)
*Fig. 4. Evaluations in experiments on off-policy sampling*

### 3.3 Bootstrapping

![bootstrapping](presentation_report/bootstrapping.png)
*Fig. 5. Evaluations in experiments on bootstrapping.*

### 3.4 Recurrent Plolicy

![policy_net](presentation_report/policy_net.png)
*Fig. 6. Replacement of the policy network with LSTM architecture. The left is the original fully-connected policy network where the activations are tanh; the right is our newly applied LSTM policy network where the activations are changed to ReLU.*

![lstm](presentation_report/lstm.png)
*Fig. 7. Evaluations in experiments on LSTM policy*



### 4. Conclusion

![summary](presentation_report/summary.png)
*Fig. 8. Summarized results of experiments on our extensions*

![gae](presentation_report/gae.png)
*Fig. 9. Performance of GAE applied with different λ value.*