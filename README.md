# AWS DeepRacer PPO Agent — Visual Reinforcement Learning Study

This repository presents a **research-oriented study** of a custom **Proximal Policy Optimization (PPO)** agent trained for **AWS DeepRacer** using **raw front-facing camera input**.

The work focuses on **policy stability, reward-driven behavior, and visual generalization** in a simulated autonomous driving environment. The emphasis is placed on **algorithmic understanding and empirical analysis**, rather than competition performance or deployment.

---

## Trained Policy Demonstration

The animation below shows a **fully trained PPO policy** operating in a **time-trial setting** on the `reInvent2019_wide` track.

<p align="center">
  <img src="demos/reInvent2019_wide-time_trial-my_fancy_agent-episode-0-ezgif.com-video-to-gif-converter.gif" width="420">
</p>

**What is shown:**
- a converged PPO policy (no exploration noise)
- stable lane-centering behavior
- smooth steering through long curves
- speed modulation aligned with track geometry
- low oscillation and low policy variance

This behavior corresponds to the **final policy checkpoint**, after successful completion of **five consecutive clean laps**, which is the stopping criterion used throughout this study.

---

## Full Technical Report

The full experimental report is available as a PDF:

<p align="center">
  <a href="https://github.com/AdrianKazi/AWS-DeepRacer-PPO-Agent/blob/main/AWS_DeepRacer_PPO_Agent_Architecture_and_Training.pdf" target="_blank">
    <strong>📄 View Paper</strong>
  </a>
</p>

This document includes:
- formal PPO formulation
- CNN architecture details
- reward shaping analysis
- training dynamics and failure cases
- quantitative evaluation plots

---

## Project Overview

- **Environment:** AWS DeepRacer (simulation)
- **Observation space:** Front-facing RGB camera
- **Action space:** Discrete (steering × speed)
- **Algorithm:** Proximal Policy Optimization (PPO)
- **Framework:** PyTorch
- **Primary focus:** Time-Trial on `reInvent2019_wide`

An agent is considered trained once it completes **five consecutive clean laps** without going off-track.

Tracks `reInvent2019_track` and `New_York_Track` were evaluated experimentally but did not reach stable convergence and are therefore analyzed in the report rather than emphasized visually.

---

## Model Architecture

### Visual Encoder (CNN)

The agent receives **raw pixel observations** from the front-facing camera. A convolutional neural network is used to extract spatial features relevant for driving:

- three convolutional blocks (Conv2D → BatchNorm → ReLU)
- adaptive pooling to an 8×8 spatial resolution
- shared feature extractor feeding:
  - a policy head (action logits)
  - a value head (state value estimate)

This architecture preserves spatial structure while remaining lightweight enough to avoid overfitting to visual noise.

No hand-engineered features (e.g. distance from center) are used; the policy is learned entirely from visual input.

---

### PPO Objective

PPO is selected for its robustness in environments with noisy rewards and high-dimensional observations.

The clipped surrogate objective is used:

$$
L^{CLIP}(\theta) =
\mathbb{E}_t \left[
\min\left(
r_t A_t,\,
\text{clip}(r_t, 1-\epsilon, 1+\epsilon) A_t
\right)
\right]
$$

The full loss combines policy, value, and entropy terms:

$$
L = L^{CLIP} - c_1 (V_\theta(s_t) - V_t)^2 + c_2 H(\pi_\theta)
$$



This formulation constrains policy updates while encouraging sufficient exploration.

---

## Action Space and Policy Design

The action space consists of **31 discrete actions**, each corresponding to a `(steering, speed)` pair:

- steering angles range from −30° to +30°
- speed is highest near zero steering and lower for sharper turns

This structure encodes a weak inductive bias toward:
- fast straight-line driving
- cautious cornering

Policy robustness is further improved by:
- temperature-scaled logits (T = 1.5)
- dropout (0.3) in the policy head
- explicit entropy regularization

---

## Reward Function (Time-Trial)

The reward function is shaped to encourage efficient and stable driving:

- positive reward for staying close to the center line
- speed-based bonus when aligned with the lane
- mild penalty for low-speed behavior
- implicit discouragement of oscillatory steering

The reward design provides dense feedback while avoiding explicit hard constraints.

---

## Training Strategy

Training is organized into phases, with emphasis on **Phase I**.

### Phase I — Time-Trial (Primary Result)

- training from scratch on `reInvent2019_wide`
- consistent convergence
- stable entropy decay
- reliable lap completion

### Phase II / III — Obstacles and Head-to-Head

These configurations were explored experimentally but did not fully converge:

- obstacle avoidance showed early learning signals
- head-to-head suffered from early entropy collapse

These failure modes are analyzed in the report and motivate future architectural improvements.

---

## Empirical Findings

Key observations from the experiments:

- PPO with a shallow CNN is sufficient for stable driving on wide tracks
- reward shaping has a stronger effect than network depth
- visual generalization fails on environments with significant background clutter
- entropy collapse leads to irreversible policy stagnation
- transfer learning across visually dissimilar tracks is non-trivial

---

## Repository Structure

```text
.
├── src/            # PPO agent, CNN, training logic
├── configs/        # agent and environment configurations
├── demos/          # evaluation GIFs and videos
├── plots/          # training and evaluation figures
├── README.md
├── SETUP.md
├── requirements.txt
├── Dockerfile
└── .gitignore
`
```

---

## Author
Adrian S. Kazi

Georgia Institute of Technology

Independent Research Project — Reinforcement Learning
