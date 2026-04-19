# Reinforcement Trading

A production-grade reinforcement learning system for automated forex trading on EURUSD. Built with PPO (Proximal Policy Optimization) via Stable-Baselines3, a custom Gymnasium environment, Markov chain regime detection, and a risk management layer designed for real capital deployment.

---

## Overview

This system trains an RL agent to trade EURUSD on 1-hour bars. The agent observes a rolling window of technical features, market structure indicators, and session context, and outputs discrete actions: open long, open short, close, or hold — each with a specific stop-loss and take-profit configuration.

The goal is not to build a black box. Every component is explicit: the features the agent sees, the reward it receives, the rules that can override it, and the conditions under which it is allowed to trade real money.

**Current status:** Active development. See [IMPROVEMENT_PLAN.md](./IMPROVEMENT_PLAN.md) for the full roadmap to capital readiness.

---

## Architecture

```
OHLCV Data (CSV or broker feed)
        │
        ▼
  indicators.py
  Feature Engineering
  ~36 technical + time features
        │
        ▼
  trading_env.py
  Gymnasium Environment
  Position persistence, SL/TP, spread, slippage
        │
        ▼
  train_agent.py
  PPO (Stable-Baselines3)
  Walk-forward CV, parallel envs, Sharpe-based selection
        │
        ▼
  Trained Model + Scaler + Markov Model
        │
        ▼
  risk_manager.py         ◄── Account-level safety net
  Drawdown limits, daily loss cap, lot sizing
        │
        ▼
  live_trader.py
  Hourly inference loop
        │
        ▼
  broker/oanda_broker.py
  OANDA REST API (paper or live)
```

---

## Repository Structure

```
Reinforcement-Trading/
├── config.py                 # Paths, constants, environment defaults
├── indicators.py             # Feature engineering (all technical indicators)
├── trading_env.py            # Gymnasium trading environment
├── markov_chain.py           # Markov chain market regime detection
├── risk_manager.py           # Risk rules, position sizing
├── train_agent.py            # Training script
├── evaluate.py               # Standalone evaluation + reporting
├── live_trader.py            # Live execution loop
├── health_server.py          # HTTP status endpoint
│
├── broker/
│   ├── base_broker.py        # Abstract broker interface
│   └── oanda_broker.py       # OANDA v20 adapter
│
├── data/                     # Historical OHLCV CSV files (not committed)
├── models/                   # Saved models, scaler, Markov matrix
├── checkpoints/              # Training checkpoints
├── logs/                     # Evaluation reports, trade logs
├── tensorboard_log/          # TensorBoard training metrics
├── tests/                    # Unit tests
│
├── requirements.txt
├── README.md
└── IMPROVEMENT_PLAN.md       # Full development roadmap
```

---

## Quick Start

### Prerequisites

- Python 3.10+
- CUDA-capable GPU recommended for training (CPU works but is slow)

### Installation

```bash
git clone https://github.com/untitledsolomon/Reinforcement-Trading.git
cd Reinforcement-Trading
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Data Setup

Place your EURUSD 1-hour CSV files in the `data/` directory. Expected column names:

```
Time (EET), Open, High, Low, Close, Volume
```

The `Time (EET)` column should be parseable by `pandas.to_datetime` with `dayfirst=True`.

Rename your files to match the paths in `config.py`:
- `data/train_EURUSD_1H_2020_2023.csv` — training data
- `data/test_EURUSD_1H_2023_2025.csv` — held-out test data (do not touch until final evaluation)

### Train

```bash
python train_agent.py
```

This will:
1. Load and engineer features from the training CSV.
2. Fit a StandardScaler on training data and save it to `models/scaler.pkl`.
3. Build a Markov transition matrix from training data and save it to `models/markov_model.pkl`.
4. Train a PPO agent for 10M timesteps using 8 parallel environments.
5. Evaluate checkpoints on the validation split using Sharpe ratio.
6. Save the best model to `models/best_model.zip`.

Training logs stream to `tensorboard_log/`. Launch TensorBoard to monitor:

```bash
tensorboard --logdir tensorboard_log
```

### Evaluate

```bash
python evaluate.py --model models/best_model --data data/test_EURUSD_1H_2023_2025.csv
```

Produces:
- Console output with full performance statistics
- `logs/eval_report.json` — machine-readable stats
- `logs/eval_equity_curve.png` — equity and drawdown chart

### Paper Trade

```bash
export OANDA_ACCOUNT_ID=your_practice_account_id
export OANDA_API_KEY=your_practice_api_key
python live_trader.py --model models/best_model --symbol EUR_USD --env practice
```

Do not use `--env live` until the system has passed all criteria in the [Definition of Done](#definition-of-done).

---

## Features

### Technical Indicators (indicators.py)

| Category | Features |
|---|---|
| Trend | RSI-14, MA-20 slope, MA-50 slope, MA spread, MA spread slope |
| Distance | Close–MA20 diff, Close–MA50 diff |
| Volatility | ATR-14, ATR ratio, Bollinger %B, Bollinger width |
| Momentum | MACD histogram, Stochastic %K, Stochastic %D |
| Time | Hour sin/cos, day-of-week sin/cos, London session, NY session, overlap, Asian |
| Candle | Body ratio, upper wick ratio, lower wick ratio, close position in bar |
| Returns | Log returns over 1, 3, 6, 12, 24 bars |
| Volume | Volume ratio (vs 20-bar mean), volume rate of change |

All features are normalized using a `StandardScaler` fit on the training set only.

### Environment (trading_env.py)

- **Observation:** Rolling window (30 bars) × (36 features + 3 state features + 4 Markov state probabilities)
- **Actions:** HOLD | CLOSE | OPEN (direction × SL × TP) — 20 discrete actions
- **Position model:** One position at a time, long or short
- **Costs:** Spread (1.0 pip), commission (configurable), slippage (uniform random up to 0.2 pip)
- **SL/TP:** Checked intrabar on next bar's High/Low range. Assumes SL hits first if both are touched in the same bar (conservative)
- **Reward:** Net pips on close + hold bonus (for profitable positions) − time cost − open penalty
- **Termination:** End of data, episode step limit, or 20% drawdown

### Markov Chain Regime Detection (markov_chain.py)

Four market states:

| State | Description | Typical behavior |
|---|---|---|
| `trend_up` | Rising MAs, positive momentum, RSI > 50 | Run longs, tight stops |
| `trend_down` | Falling MAs, negative momentum, RSI < 50 | Run shorts, tight stops |
| `range_high_vol` | High ATR relative to mean, mixed momentum | Reduce size or skip |
| `range_low_vol` | Low ATR, flat MAs, quiet market | Skip or very tight entries |

The transition matrix is estimated from training data. At inference time, a `MarkovRegimeFilter` maintains a belief distribution over states using Bayesian updates and passes the 4-state probability vector to the RL agent as additional observation features.

### Risk Manager (risk_manager.py)

Account-level rules applied independently of the model:

| Rule | Default | Description |
|---|---|---|
| Daily drawdown cap | 5% | Blocks new trades if daily loss exceeds 5% of equity |
| Total drawdown cap | 15% | Halts system if drawdown from peak exceeds 15% |
| Max trades per day | 10 | Prevents overtrading / runaway loops |
| Consecutive loss limit | 5 | Forces a cool-off after 5 straight losses |
| Minimum equity | $5,000 | Hard stop — never trade below this level |

Position sizing uses fixed fractional method: `lot_size = (equity × risk_pct) / (sl_pips × pip_value)`. Default: 1% equity risk per trade.

---

## Training Details

| Parameter | Value | Notes |
|---|---|---|
| Algorithm | PPO | Stable-Baselines3 v2.5+ |
| Policy | MlpPolicy | 2× [256, 256] layers for actor and critic |
| Total timesteps | 10,000,000 | ~14 hours on a modern GPU with 8 parallel envs |
| Parallel environments | 8 | `SubprocVecEnv` |
| Observation window | 30 bars | ~30 hours of price history per step |
| Action space | 20 discrete | 2 directions × 3 SL × 3 TP + 2 (hold/close) |
| Validation | Walk-forward, 4 windows | Model selected by mean Sharpe across all windows |
| Checkpoint frequency | Every 500k steps | Best checkpoint saved automatically by `EvalCallback` |

### Hyperparameters

```python
PPO(
    learning_rate = 3e-4,
    n_steps       = 2048,
    batch_size    = 256,
    n_epochs      = 10,
    gamma         = 0.99,
    gae_lambda    = 0.95,
    clip_range    = 0.2,
    ent_coef      = 0.01,
    vf_coef       = 0.5,
    max_grad_norm = 0.5,
)
```

---

## Evaluation Metrics

A model is considered for deployment only when it achieves all of the following on the held-out test dataset:

| Metric | Minimum threshold |
|---|---|
| Sharpe Ratio (annualized) | ≥ 1.5 |
| Maximum Drawdown | ≤ 15% |
| Win Rate | ≥ 40% |
| Profit Factor | ≥ 1.3 |
| Walk-forward pass rate | 3 of 4 windows |
| Paper trading period | ≥ 90 days |

---

## Live Trading

The live trading loop (`live_trader.py`) runs on the H1 bar close. It:

1. Fetches the latest 300 bars from OANDA.
2. Computes all features using the same pipeline as training.
3. Normalizes using the saved scaler.
4. Queries the Markov filter for the current regime.
5. Gets a model action.
6. Passes the proposed action through the risk manager.
7. If approved, submits to OANDA via the REST API with SL and TP orders attached.
8. Sleeps until the next bar close.

### Environment Variables (required for live trading)

```bash
OANDA_ACCOUNT_ID=your_account_id
OANDA_API_KEY=your_api_token
```

### Health Monitoring

```bash
# In a separate terminal or systemd service:
python health_server.py

# Check status:
curl http://localhost:8080/health    # 200 = running, 503 = halted
curl http://localhost:8080/stats     # JSON with equity, drawdown, last action
```

---

## Development Roadmap

See [IMPROVEMENT_PLAN.md](./IMPROVEMENT_PLAN.md) for the full phase-by-phase plan with exact implementation details.

**Summary of phases:**

| Phase | Description | Status |
|---|---|---|
| 1 | Foundation fixes (normalization, action space, reward shaping, test script) | In progress |
| 2 | Feature engineering overhaul (time, candle, momentum, volume features) | Planned |
| 3 | Reward function redesign | Planned |
| 4 | Training infrastructure (walk-forward, parallel envs, 10M steps) | Planned |
| 5 | Model evaluation and selection (Sharpe-based, full report) | Planned |
| 6 | Markov chain regime detection integration | Planned |
| 7 | Risk management layer | Planned |
| 8 | Live trading infrastructure (broker adapter, execution loop) | Planned |
| 9 | Monitoring and observability (logging, health endpoint, TensorBoard) | Planned |

---

## Requirements

Key dependencies (see `requirements.txt` for full list with pinned versions):

```
stable-baselines3>=2.5.0
gymnasium>=1.0.0
torch>=2.0.0
pandas>=2.0.0
pandas-ta>=0.3.14b0
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
optuna>=3.0.0          # Optional: hyperparameter tuning
oandapyV20>=0.7.0      # Optional: live trading only
tensorboard>=2.13.0
```

---

## Disclaimers

This system is for research and educational purposes. Automated trading involves substantial risk of loss. Past performance in backtesting or paper trading does not guarantee future results. Never trade with money you cannot afford to lose. Always start with a paper trading account and a small capital allocation.

The Markov chain regime detection model is statistical and will misclassify market states, especially during unusual macro events. The risk manager is a safety net, not a guarantee.

---

## License

MIT License. See `LICENSE` for details.

---

## Contributing

1. Read `IMPROVEMENT_PLAN.md` before starting any work.
2. Complete phases in order — do not start Phase 3 before Phase 1 is done.
3. Every new feature must have a corresponding test in `tests/`.
4. Do not commit trained model files or API keys.
5. Run `python -m pytest tests/` before opening a pull request.
