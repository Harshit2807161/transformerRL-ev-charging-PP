# Optimizing In-Home EV Charging with Transformers and Policy-Based DRL

## Abstract

Managing EV charging is challenging due to factors like limited battery capacity, unpredictable user behavior, and fluctuating electricity prices. This project focuses on optimizing in-home EV charging using a deep reinforcement learning (DRL)-based Markov decision process (MDP) framework integrated with transformer-based models for forecasting.

We propose a time-series transformer-based network for electricity price forecasting, enhancing the scheduling of EV charging to minimize costs and maximize user satisfaction. Our transformer-based feature extraction model investigates historical price patterns over different time frames. DRL techniques like Deep Q-Networks (DQN), Deep Deterministic Policy Gradient (DDPG), and Proximal Policy Optimization (PPO) are used for decision-making, and we compare the results using three transformers: Autoformer, Informer, and PatchTST.

Our approach significantly reduces EV charging costs while maintaining high user satisfaction, outperforming prior models. The results show a 125.74% reduction in costs for continuous action space and 140.66% in discrete action space.

## Key Features

- **Deep Reinforcement Learning (DRL) Models:** Includes DQN, DDPG, and PPO models in both continuous and discrete forms.
- **Transformer Models:** Utilizes transformer-based models (Autoformer, Informer, PatchTST) to capture electricity price patterns and improve forecasts.
- **Time-series forecasting:** Employs different time-frames (past 24 hours, 24 days, and 24 weeks) for forecasting electricity prices.
- **Decision-making Models:** The decision framework determines optimal charging or discharging actions based on the predicted electricity price and the state of charge (SoC).

## Repository Structure

```plaintext
.
├── data/
│   ├── electricity_prices.csv  # Historical electricity price data used for training
│   ├── household_data.csv      # Simulated household electricity consumption data
│   └── ...
├── models/
│   ├── dqn.py                 # Implementation of Deep Q-Network
│   ├── ddpg.py                # Implementation of Deep Deterministic Policy Gradient
│   ├── ppo.py                 # Implementation of Proximal Policy Optimization
│   ├── autoformer.py          # Autoformer model for price prediction
│   ├── informer.py            # Informer model for time-series analysis
│   └── patchtst.py            # PatchTST model for feature extraction
├── training/
│   ├── train_transformer.py   # Script to train transformer-based models on price data
│   ├── train_drl.py           # Script to train DRL models with transformers
│   └── evaluation.py          # Evaluation scripts for trained models
├── results/
│   ├── training_logs/         # Logs for model training
│   └── evaluation_results/    # Results and metrics from the evaluation of models
├── figures/
│   ├── transformer_architecture.png  # Transformer model architecture
│   ├── ev_charging_framework.png     # EV charging and discharging scheduling framework
│   ├── forecast_comparison.png       # Comparison of electricity price forecasts
│   ├── training_progress.png         # Training progress of DRL models
│   └── results_plots/                # Plots showcasing the results of simulations
└── README.md
```

## Installation

To run the project, clone the repository and install the necessary dependencies:

```bash
git clone https://github.com/your-username/ev-charging-optimization.git
cd ev-charging-optimization
pip install -r requirements.txt
```

Dependencies can be found in the `requirements.txt` file, and they include popular machine learning libraries such as TensorFlow or PyTorch, transformers, and DRL frameworks.

## Usage

### 1. Data Preparation

Ensure the electricity price data and household electricity consumption data are placed in the `data/` directory. The dataset used in this project consists of historical price data for different time periods, including:

- Past 24 hours
- Same hour over the last 24 days
- Same hour on the same weekday over the last 24 weeks

### 2. Model Training

#### Transformer Model Training
Run the following script to train the transformer-based models (Autoformer, Informer, PatchTST) for electricity price forecasting:

```bash
python training/train_transformer.py --model autoformer
```

#### Deep Reinforcement Learning (DRL) Training
Use the following command to train the DRL models (DQN, DDPG, PPO) on top of the transformer-based predictions:

```bash
python training/train_drl.py --model ppo --price_model autoformer
```
