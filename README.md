Here’s the updated `README.md` with the correct placeholders for the images in the appropriate sections. You can replace the paths in `figures/` once the images are added.

---

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

### 3. Evaluation

After training, evaluate the models using the `evaluation.py` script:

```bash
python training/evaluation.py --model ppo --price_model autoformer
```

## Figures

### Transformer Model Architecture

This diagram illustrates the architecture of the transformer model used for electricity price prediction. The model captures long-term dependencies through attention mechanisms to make more accurate price forecasts.

![Transformer Architecture](figures/transformer_architecture.png)

### EV Charging and Discharging Scheduling Framework

The proposed framework for EV charging and discharging optimization combines transformer-based forecasting and DRL models (DQN, DDPG, PPO) to minimize charging costs.

![EV Charging Framework](figures/ev_charging_framework.png)

### Forecast Comparison

Comparison of electricity price forecasts generated by Autoformer, Informer, and PatchTST models, showing the accuracy of each model over the given time frames.

![Forecast Comparison](figures/forecast_comparison.png)

### Training Progress

A graphical representation of the training progress of the DRL models over time, tracking the performance improvements and rewards achieved during training.

![Training Progress](figures/training_progress.png)

### Results

The results of our simulations demonstrate significant reductions in EV charging costs, presented in terms of percentage savings achieved by our model in both continuous and discrete action spaces.

![Results Plots](figures/results_plots/savings_comparison.png)

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

---

You can update the image paths with the actual locations after placing the image files into the `figures/` directory.

