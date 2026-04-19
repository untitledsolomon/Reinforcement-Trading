# Reinforcement Trading Bot — Full Improvement Plan
### For AI Agents (Jules, Copilot, etc.) and Human Contributors

> **Purpose:** This document is the single source of truth for bringing this RL trading system from its current research prototype state to a capital-ready, production-grade automated trading system. Every task is described with enough precision that an AI coding agent can implement it without guessing. Do not skip phases. Each phase builds on the previous one.

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Current State Audit](#current-state-audit)
3. [Phase 1 — Foundation Fixes (Critical, Do First)](#phase-1--foundation-fixes)
4. [Phase 2 — Feature Engineering Overhaul](#phase-2--feature-engineering-overhaul)
5. [Phase 3 — Reward Function Redesign](#phase-3--reward-function-redesign)
6. [Phase 4 — Training Infrastructure](#phase-4--training-infrastructure)
7. [Phase 5 — Model Evaluation & Selection](#phase-5--model-evaluation--selection)
8. [Phase 6 — Markov Chain Integration](#phase-6--markov-chain-integration)
9. [Phase 7 — Risk Management Layer](#phase-7--risk-management-layer)
10. [Phase 8 — Live Trading Infrastructure](#phase-8--live-trading-infrastructure)
11. [Phase 9 — Monitoring & Observability](#phase-9--monitoring--observability)
12. [Nice-to-Haves & Future Enhancements](#nice-to-haves--future-enhancements)
13. [File Structure (Target)](#file-structure-target)
14. [Definition of Done](#definition-of-done)

---

## System Overview

**Bot type:** Reinforcement Learning (RL) Forex trading agent using PPO (Proximal Policy Optimization) via Stable-Baselines3.

**Target instrument:** EURUSD (expandable to other pairs).

**Timeframe:** 1-hour bars (primary). 15-minute and 4-hour as auxiliary context (future).

**Core loop:**
```
Raw OHLCV data
    → Feature engineering (indicators.py)
    → Normalization
    → RL Environment (trading_env.py)
    → PPO Agent (train_agent.py)
    → Evaluation & checkpoint selection
    → Risk-gated live execution
```

**Capital readiness definition:** The system is capital-ready when it demonstrates a Sharpe Ratio ≥ 1.5, maximum drawdown ≤ 15%, and positive expectancy over at least 6 months of out-of-sample data across at least 2 walk-forward windows.

---

## Current State Audit

The following issues exist in the current codebase and must be resolved. This section is fact, not opinion — derived from direct inspection of the source files.

### `indicators.py`
- Only 8 features: RSI, ATR, two MA slopes, two MA diffs, MA spread, MA spread slope.
- No time-of-day or day-of-week features. The agent cannot distinguish a London session open from an Asian session close.
- No candle structure features (body/wick ratios).
- No multi-timeframe returns.
- No volume-relative features.
- No MACD, Bollinger Bands, or stochastic.

### `trading_env.py`
- Normalization parameters (`feature_mean`, `feature_std`) are accepted but never actually passed in any instantiation call — raw unscaled features go into the neural network.
- State features `time_in_trade` is normalized by dividing by 1000, `unrealized_pips` by 100 — these are hardcoded magic numbers with no justification.
- The observation window pads missing bars by repeating the first row — this is a valid but suboptimal approach. Should use zero-padding or the window mean instead.
- Single lot size hardcoded at 100,000 units. Position sizing is binary (in or out), not continuous.
- No account drawdown termination — the episode continues even if equity goes to zero.
- No maximum position duration enforcement beyond the global episode cap.

### `train_agent.py`
- **All reward shaping is zeroed out:** `hold_reward_weight=0.0`, `open_penalty_pips=0.0`, `time_penalty_pips=0.0`, `unrealized_delta_weight=0.0`. The agent only receives reward on closed trades, creating extreme reward sparsity.
- Action space: 2 directions × 8 SL options × 8 TP options + 2 = **130 actions**. This is excessive and makes policy learning unstable.
- Training timesteps: 600,000. Insufficient for a 130-action, 240-dimensional observation space problem.
- Model selection criterion: final equity (a single scalar). This is noisy and gameable by lucky single trades.
- Uses `DummyVecEnv` (single environment). No parallelism.
- No hyperparameter tuning.
- The data file path references a file that does not exist in the repository (`data/EURUSD_Hourly_Ask_2015.12.01_2025.12.16.csv`).

### `test_agent.py`
- References a non-existent data file path.
- Applies an 80/20 split on the test file itself — this is incorrect; the test script should use a separate, pre-designated test dataset only.
- No statistical analysis of trade results (win rate, average RR, max drawdown, Sharpe, Calmar).
- No comparison against a buy-and-hold benchmark.

---

## Phase 1 — Foundation Fixes

**Goal:** Get the system running correctly with no silent bugs. These are blocking issues.

---

### Task 1.1 — Standardize the data directory and file paths

**File:** `train_agent.py`, `test_agent.py`, project root.

**Problem:** Both scripts reference absolute or non-existent file paths. Any contributor cloning the repo cannot run the code without manually editing paths.

**Implementation:**

1. Create a `data/` directory in the project root.
2. Place the following files in `data/`:
   - `train_EURUSD_1H_2020_2023.csv` — training data
   - `test_EURUSD_1H_2023_2025.csv` — held-out test data (never used in training or validation)
3. Create `config.py` in the project root:

```python
# config.py
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
LOG_DIR   = os.path.join(BASE_DIR, "logs")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")

TRAIN_DATA_PATH = os.path.join(DATA_DIR, "train_EURUSD_1H_2020_2023.csv")
TEST_DATA_PATH  = os.path.join(DATA_DIR, "test_EURUSD_1H_2023_2025.csv")

# Environment defaults
DEFAULT_WINDOW_SIZE   = 30
DEFAULT_SL_OPTIONS    = [15, 30, 60]
DEFAULT_TP_OPTIONS    = [15, 30, 60]
DEFAULT_SPREAD_PIPS   = 1.0
DEFAULT_SLIPPAGE_PIPS = 0.2
DEFAULT_LOT_SIZE      = 100_000.0
INITIAL_EQUITY_USD    = 10_000.0
```

4. Update all file path references in `train_agent.py` and `test_agent.py` to import from `config.py`.

---

### Task 1.2 — Implement and apply feature normalization

**Files:** `indicators.py`, `trading_env.py`, `train_agent.py`, `test_agent.py`.

**Problem:** Raw feature values with vastly different scales (RSI: 0–100, ATR: 0.0002–0.002, MA diffs: ±0.005) feed into the neural network without normalization. This causes gradient instability and slows learning.

**Implementation:**

In `indicators.py`, add a normalization helper at the bottom of `load_and_preprocess_data`:

```python
from sklearn.preprocessing import StandardScaler
import numpy as np

def fit_scaler(df: pd.DataFrame, feature_cols: list):
    """Fit a StandardScaler on the provided feature columns. Returns scaler."""
    scaler = StandardScaler()
    scaler.fit(df[feature_cols].values)
    return scaler

def apply_scaler(df: pd.DataFrame, feature_cols: list, scaler) -> pd.DataFrame:
    """Apply a pre-fitted scaler to a dataframe. Returns a copy."""
    df = df.copy()
    df[feature_cols] = scaler.transform(df[feature_cols].values)
    return df
```

In `train_agent.py`, fit the scaler on training data only and pass the parameters to all environments:

```python
from indicators import load_and_preprocess_data, fit_scaler
import numpy as np

df, feature_cols = load_and_preprocess_data(TRAIN_DATA_PATH)
split_idx = int(len(df) * 0.8)
train_df = df.iloc[:split_idx].copy()
val_df   = df.iloc[split_idx:].copy()

# Fit ONLY on training data
scaler = fit_scaler(train_df, feature_cols)
feature_mean = scaler.mean_.astype(np.float32)
feature_std  = scaler.scale_.astype(np.float32)

# Pass to every environment factory
def make_train_env():
    return ForexTradingEnv(
        df=train_df,
        feature_columns=feature_cols,
        feature_mean=feature_mean,
        feature_std=feature_std,
        ...
    )
```

**Critical rule:** The scaler must be fit on training data only. Fitting on the full dataset is data leakage. Save the scaler to disk alongside the model so it can be reloaded for live inference:

```python
import pickle
with open("models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
```

---

### Task 1.3 — Reduce and rationalize the action space

**File:** `train_agent.py`, `trading_env.py`.

**Problem:** 130 discrete actions is too many. PPO's policy network outputs a categorical distribution over all actions. With 130 options, the agent must learn to distinguish between SL=90 and SL=120 at the same time it learns when to enter the market — this slows convergence dramatically.

**Implementation:**

Replace the SL/TP option lists in `train_agent.py`:

```python
# Before (130 actions):
SL_OPTS = [5, 10, 15, 25, 30, 60, 90, 120]
TP_OPTS = [5, 10, 15, 25, 30, 60, 90, 120]

# After (20 actions: 2 directions × 3 SL × 3 TP + 2 = 20):
SL_OPTS = [15, 30, 60]
TP_OPTS = [15, 30, 60]
```

**Rationale for these specific values:**
- 15 pips: tight, scalp-style stop suitable for low-volatility consolidation
- 30 pips: medium stop, approximately 1 ATR on EURUSD 1H
- 60 pips: wide stop for trend-following trades

If you want to test a continuous action space in the future (Phase 4, Task 4.4), use SAC instead of PPO and treat SL/TP as continuous outputs clipped to [5, 120].

---

### Task 1.4 — Fix the test script

**File:** `test_agent.py`

**Problems:**
1. Applies a train/test split on the test CSV — this is wrong. The test script should use the dedicated test file from `config.py`.
2. No statistical output beyond an equity curve.

**Implementation:**

Replace the file loading and split logic entirely:

```python
from config import TEST_DATA_PATH, MODEL_DIR
import pickle, os

df, feature_cols = load_and_preprocess_data(TEST_DATA_PATH)
test_df = df.copy()  # Use the entire test file — no further splitting

# Load the scaler that was saved during training
with open(os.path.join(MODEL_DIR, "scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)

feature_mean = scaler.mean_.astype(np.float32)
feature_std  = scaler.scale_.astype(np.float32)
```

Add a statistics function after `run_one_episode`:

```python
def compute_statistics(equity_curve: list, closed_trades: list, initial_equity: float = 10000.0):
    import numpy as np

    equity = np.array(equity_curve)
    returns = np.diff(equity) / equity[:-1]

    # Sharpe (annualized, assuming 1H bars, 6500 trading hours/year)
    sharpe = (returns.mean() / (returns.std() + 1e-9)) * np.sqrt(6500)

    # Max drawdown
    peak = np.maximum.accumulate(equity)
    drawdown = (peak - equity) / peak
    max_dd = drawdown.max()

    # Calmar ratio
    total_return = (equity[-1] - initial_equity) / initial_equity
    calmar = total_return / (max_dd + 1e-9)

    # Trade stats
    if closed_trades:
        wins   = [t for t in closed_trades if t["net_pips"] > 0]
        losses = [t for t in closed_trades if t["net_pips"] <= 0]
        win_rate  = len(wins) / len(closed_trades)
        avg_win   = np.mean([t["net_pips"] for t in wins])   if wins   else 0
        avg_loss  = np.mean([t["net_pips"] for t in losses]) if losses else 0
        avg_rr    = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        expectancy = win_rate * avg_win + (1 - win_rate) * avg_loss
    else:
        win_rate = avg_win = avg_loss = avg_rr = expectancy = 0

    return {
        "final_equity":  float(equity[-1]),
        "total_return":  float(total_return),
        "sharpe":        float(sharpe),
        "max_drawdown":  float(max_dd),
        "calmar":        float(calmar),
        "num_trades":    len(closed_trades),
        "win_rate":      float(win_rate),
        "avg_win_pips":  float(avg_win),
        "avg_loss_pips": float(avg_loss),
        "avg_rr":        float(avg_rr),
        "expectancy":    float(expectancy),
    }
```

Print the full statistics dictionary at the end of `main()` and also save it to `logs/test_stats.json`.

---

### Task 1.5 — Add drawdown-based episode termination

**File:** `trading_env.py`

**Problem:** The environment does not terminate when equity is destroyed. An agent can blow the account and keep receiving steps. This teaches the agent that blowing up has no consequences.

**Implementation:**

Add a `max_drawdown_pct` parameter to `ForexTradingEnv.__init__`:

```python
def __init__(self, ..., max_drawdown_pct: float = 0.20):
    ...
    self.max_drawdown_pct = float(max_drawdown_pct)
    self.peak_equity = self.initial_equity_usd
```

In `_reset_state`, add:
```python
self.peak_equity = self.initial_equity_usd
```

In `step`, after updating equity, add:
```python
# Update peak equity
if self.equity_usd > self.peak_equity:
    self.peak_equity = self.equity_usd

# Terminate if drawdown exceeds threshold
current_dd = (self.peak_equity - self.equity_usd) / self.peak_equity
if current_dd >= self.max_drawdown_pct:
    self.terminated = True
    reward_pips -= 50.0  # Large penalty for blowing the account
```

Use `max_drawdown_pct=0.20` for training (terminate at 20% drawdown) and `max_drawdown_pct=0.15` for live deployment.

---

## Phase 2 — Feature Engineering Overhaul

**Goal:** Give the agent a significantly richer view of market state so it can learn meaningful patterns.

---

### Task 2.1 — Add time-of-day and day-of-week features

**File:** `indicators.py`

**Problem:** EURUSD behaves very differently by session. London open (07:00–09:00 UTC) has high directional momentum. Asian session (00:00–07:00 UTC) is low volatility and range-bound. The agent cannot currently distinguish these contexts.

**Implementation:**

Add to `load_and_preprocess_data`, after the existing indicators:

```python
import numpy as np

# Time features — encoded as sin/cos pairs to handle cyclical wrap-around
# (23:00 and 00:00 should be "close" to each other — raw integer encoding breaks this)
hour = df.index.hour.astype(float)
dow  = df.index.dayofweek.astype(float)  # 0=Monday, 4=Friday

df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
df["dow_sin"]  = np.sin(2 * np.pi * dow / 7)
df["dow_cos"]  = np.cos(2 * np.pi * dow / 7)

# Session flags (binary)
# London: 06:00–16:00 UTC, New York: 12:00–21:00 UTC
df["session_london"]   = ((hour >= 6)  & (hour < 16)).astype(float)
df["session_ny"]       = ((hour >= 12) & (hour < 21)).astype(float)
df["session_overlap"]  = ((hour >= 12) & (hour < 16)).astype(float)  # London/NY overlap — highest volatility
df["session_asian"]    = ((hour >= 21) | (hour < 6)).astype(float)
```

Add all 8 new columns to `feature_cols`.

---

### Task 2.2 — Add candle structure features

**File:** `indicators.py`

**Problem:** The agent sees only derived indicators, not the raw price action structure of each bar. Candle body/wick ratios are among the most information-dense features in technical analysis.

**Implementation:**

```python
bar_range = df["High"] - df["Low"] + 1e-9  # avoid division by zero

# Body ratio: positive = bullish bar, negative = bearish bar. Range: [-1, 1]
df["body_ratio"] = (df["Close"] - df["Open"]) / bar_range

# Upper wick as fraction of total bar range. Range: [0, 1]
df["upper_wick_ratio"] = (df["High"] - df[["Open", "Close"]].max(axis=1)) / bar_range

# Lower wick as fraction of total bar range. Range: [0, 1]
df["lower_wick_ratio"] = (df[["Open", "Close"]].min(axis=1) - df["Low"]) / bar_range

# Close position within bar: 0 = closed at low, 1 = closed at high
df["close_position"] = (df["Close"] - df["Low"]) / bar_range
```

Add all 4 columns to `feature_cols`.

---

### Task 2.3 — Add multi-lookback price returns

**File:** `indicators.py`

**Problem:** The agent has no direct view of recent price momentum at multiple timescales. MA slopes give some of this, but log returns are a cleaner, scale-invariant momentum signal.

**Implementation:**

```python
for n in [1, 3, 6, 12, 24]:
    df[f"log_ret_{n}"] = np.log(df["Close"] / df["Close"].shift(n))
```

Add all 5 columns to `feature_cols`. Note: these will produce NaN for the first 24 bars, which is handled by the existing `df.dropna()` call.

---

### Task 2.4 — Add volatility and momentum indicators

**File:** `indicators.py`

**Implementation:**

```python
# Bollinger Bands
bbands = ta.bbands(df["Close"], length=20, std=2)
df["bb_pct"] = (df["Close"] - bbands["BBL_20_2.0"]) / (bbands["BBU_20_2.0"] - bbands["BBL_20_2.0"] + 1e-9)
df["bb_width"] = (bbands["BBU_20_2.0"] - bbands["BBL_20_2.0"]) / bbands["BBM_20_2.0"]  # normalized width

# MACD histogram (captures momentum shifts, not absolute level)
macd = ta.macd(df["Close"], fast=12, slow=26, signal=9)
df["macd_hist"] = macd["MACDh_12_26_9"]

# Stochastic %K and %D
stoch = ta.stoch(df["High"], df["Low"], df["Close"], k=14, d=3)
df["stoch_k"] = stoch["STOCHk_14_3_3"]
df["stoch_d"] = stoch["STOCHd_14_3_3"]

# ATR ratio: current ATR relative to its own 50-bar average (normalized volatility regime)
df["atr_ratio"] = df["atr_14"] / df["atr_14"].rolling(50).mean()
```

Add `bb_pct`, `bb_width`, `macd_hist`, `stoch_k`, `stoch_d`, `atr_ratio` to `feature_cols`.

---

### Task 2.5 — Add volume features

**File:** `indicators.py`

**Implementation:**

```python
# Volume relative to its 20-bar rolling mean (detects institutional activity)
df["vol_ratio"] = df["Volume"] / (df["Volume"].rolling(20).mean() + 1e-9)

# Volume rate of change
df["vol_roc"] = df["Volume"].pct_change(3)
```

Add `vol_ratio`, `vol_roc` to `feature_cols`.

---

### Task 2.6 — Final feature list and count check

After all additions, `feature_cols` should contain the following (approximately 36 features):

```
Original 8:  rsi_14, atr_14, ma_20_slope, ma_50_slope, close_ma20_diff, close_ma50_diff, ma_spread, ma_spread_slope
Time 8:      hour_sin, hour_cos, dow_sin, dow_cos, session_london, session_ny, session_overlap, session_asian
Candle 4:    body_ratio, upper_wick_ratio, lower_wick_ratio, close_position
Returns 5:   log_ret_1, log_ret_3, log_ret_6, log_ret_12, log_ret_24
Volatility 6: bb_pct, bb_width, macd_hist, stoch_k, stoch_d, atr_ratio
Volume 2:    vol_ratio, vol_roc
State (env): position, time_in_trade_norm, unrealized_pips_scaled  [added by trading_env.py]
```

Total observation dimension: 36 features × 30 window = 1080 per step.

---

## Phase 3 — Reward Function Redesign

**Goal:** Give the agent a dense, well-shaped reward signal that teaches it to trade profitably without overtrading or holding losing positions indefinitely.

---

### Task 3.1 — Re-enable and tune reward shaping parameters

**File:** `train_agent.py`

The following parameters in the training env must be non-zero:

```python
def make_train_env():
    return ForexTradingEnv(
        ...
        hold_reward_weight=0.005,    # Small bonus per bar while unrealized PnL > 0. Teaches "let winners run".
        open_penalty_pips=0.3,       # Penalty per trade opened. Discourages overtrading.
        time_penalty_pips=0.01,      # Cost per bar while in a trade. Discourages infinite holding.
        unrealized_delta_weight=0.0, # Keep at 0 — delta unrealized is too noisy for shaping.
    )
```

**Rationale for each value:**
- `hold_reward_weight=0.005`: At 20 pips unrealized, this gives +0.1 pips/bar bonus — small enough not to dominate the terminal reward but enough to discourage premature closes.
- `open_penalty_pips=0.3`: Forces the agent to have conviction before opening. A trade that earns less than 0.3 pips is a net negative — filters noise trades.
- `time_penalty_pips=0.01`: At 100 bars held, this accumulates to 1 pip of cost — manageable for a winning trade, punishing for a losing one.

---

### Task 3.2 — Add Sharpe-like reward normalization (optional enhancement)

**File:** `trading_env.py`

For more advanced training, replace raw pip rewards with a reward normalized by recent volatility. This trains the agent to maximize risk-adjusted returns, not raw returns.

Add to `ForexTradingEnv.__init__`:

```python
self._recent_rewards = []
self._reward_window  = 200
```

Replace the final reward line in `step`:
```python
# Current:
reward = float(reward_pips) * self.reward_scale

# Enhanced:
if reward_pips != 0:
    self._recent_rewards.append(reward_pips)
    if len(self._recent_rewards) > self._reward_window:
        self._recent_rewards.pop(0)

if len(self._recent_rewards) >= 20:
    std = np.std(self._recent_rewards) + 1e-6
    reward = float(reward_pips) / std * self.reward_scale
else:
    reward = float(reward_pips) * self.reward_scale
```

This is a quality-of-life improvement, not a requirement for Phase 3. Implement after basic reward shaping is confirmed working.

---

## Phase 4 — Training Infrastructure

**Goal:** Train the agent properly — enough steps, with parallelism, on correct data splits.

---

### Task 4.1 — Implement walk-forward cross-validation

**File:** `train_agent.py`

**Problem:** The current 80/20 time split gives a single validation result. A single OOS evaluation can be lucky or unlucky depending on market regime. Walk-forward validation tests the model across multiple market regimes.

**Implementation:**

Add a `walk_forward_splits` function:

```python
def walk_forward_splits(df, n_splits=4, train_ratio=0.7, gap_bars=0):
    """
    Generates (train_df, val_df) pairs using expanding window walk-forward.
    gap_bars: number of bars to skip between train end and val start (prevents data leakage at boundaries).
    """
    n = len(df)
    min_train = int(n * train_ratio / n_splits)
    splits = []

    for i in range(1, n_splits + 1):
        train_end = int(n * (train_ratio * i / n_splits))
        val_start = train_end + gap_bars
        val_end   = val_start + int(n * (1 - train_ratio) / n_splits)
        if val_end > n:
            val_end = n
        if val_start >= val_end:
            continue
        splits.append((
            df.iloc[:train_end].copy(),
            df.iloc[val_start:val_end].copy()
        ))

    return splits
```

Use this in training to evaluate on multiple windows and average the Sharpe ratios. A model is only considered for deployment if it achieves Sharpe ≥ 1.0 on at least 3 of 4 walk-forward windows.

---

### Task 4.2 — Use parallel environments for training

**File:** `train_agent.py`

Replace `DummyVecEnv` with `SubprocVecEnv` for the training environment. This runs N environments in parallel processes and multiplies effective sample throughput.

```python
from stable_baselines3.common.vec_env import SubprocVecEnv

N_ENVS = 8  # Set to number of CPU cores - 1

train_vec_env = SubprocVecEnv([make_train_env] * N_ENVS)
```

**Note:** `SubprocVecEnv` uses Python multiprocessing. Lambda functions are not picklable. Replace lambda factories with named functions:

```python
# Wrong (not picklable):
train_vec_env = SubprocVecEnv([lambda: ForexTradingEnv(...)] * 8)

# Correct (named function, picklable):
def make_train_env():
    return ForexTradingEnv(...)

train_vec_env = SubprocVecEnv([make_train_env] * 8)
```

---

### Task 4.3 — Increase training timesteps and add evaluation callback

**File:** `train_agent.py`

```python
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

# Minimum: 5M. Target: 10M+
TOTAL_TIMESTEPS = 10_000_000

# Evaluate on validation env every 100k steps, save best model automatically
eval_callback = EvalCallback(
    eval_env=val_vec_env,
    best_model_save_path="./models/",
    log_path="./logs/",
    eval_freq=100_000 // N_ENVS,  # Divide by n_envs because SB3 counts total env steps
    n_eval_episodes=1,
    deterministic=True,
    verbose=1,
)

checkpoint_callback = CheckpointCallback(
    save_freq=500_000 // N_ENVS,
    save_path="./checkpoints/",
    name_prefix="ppo_eurusd",
)

model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=[eval_callback, checkpoint_callback],
)
```

---

### Task 4.4 — Tune PPO hyperparameters

**File:** `train_agent.py`

The default PPO hyperparameters are not optimal for financial time-series. Use the following as a starting point:

```python
model = PPO(
    policy="MlpPolicy",
    env=train_vec_env,
    learning_rate=3e-4,          # Standard. Can try 1e-4 to 5e-4.
    n_steps=2048,                # Steps per rollout per environment
    batch_size=256,              # Mini-batch size for gradient updates
    n_epochs=10,                 # Passes over rollout data
    gamma=0.99,                  # Discount factor. High = values long-term rewards.
    gae_lambda=0.95,             # Bias-variance tradeoff in advantage estimation
    clip_range=0.2,              # PPO clip parameter
    ent_coef=0.01,               # Entropy bonus — encourages exploration
    vf_coef=0.5,                 # Value function loss weight
    max_grad_norm=0.5,           # Gradient clipping
    policy_kwargs=dict(
        net_arch=[dict(pi=[256, 256], vf=[256, 256])]  # Larger network for complex observation
    ),
    verbose=1,
    tensorboard_log="./tensorboard_log/",
)
```

**Optional — automated hyperparameter search with Optuna:**

```python
# install: pip install optuna
import optuna

def objective(trial):
    lr       = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    n_steps  = trial.suggest_categorical("n_steps", [512, 1024, 2048])
    ent_coef = trial.suggest_float("ent_coef", 0.0, 0.05)

    model = PPO("MlpPolicy", train_vec_env, learning_rate=lr, n_steps=n_steps, ent_coef=ent_coef)
    model.learn(1_000_000)
    _, sharpe = evaluate_model_sharpe(model, val_vec_env)
    return sharpe

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)
print(study.best_params)
```

---

## Phase 5 — Model Evaluation & Selection

**Goal:** Select models based on risk-adjusted performance, not final equity.

---

### Task 5.1 — Replace final-equity model selection with Sharpe-based selection

**File:** `train_agent.py`

Replace the `evaluate_model` function:

```python
def evaluate_model_sharpe(model, vec_env, n_episodes=3, deterministic=True):
    """
    Runs n_episodes and returns (mean_sharpe, mean_final_equity).
    Multiple episodes are needed because random_start means each episode
    covers a different slice of the data.
    """
    import numpy as np

    all_sharpes = []
    all_finals  = []

    for _ in range(n_episodes):
        obs = vec_env.reset()
        equity_curve = []

        while True:
            action, _ = model.predict(obs, deterministic=deterministic)
            step_out = vec_env.step(action)
            obs, _, dones, infos = step_out[:4]
            eq = infos[0].get("equity_usd", vec_env.get_attr("equity_usd")[0])
            equity_curve.append(float(eq))
            if bool(dones[0]):
                break

        equity = np.array(equity_curve)
        returns = np.diff(equity) / (equity[:-1] + 1e-9)
        sharpe = (returns.mean() / (returns.std() + 1e-9)) * np.sqrt(6500)
        all_sharpes.append(float(sharpe))
        all_finals.append(float(equity[-1]))

    return float(np.mean(all_sharpes)), float(np.mean(all_finals))
```

When iterating over checkpoints, rank by `mean_sharpe`, not `final_equity`.

---

### Task 5.2 — Generate a full evaluation report

**File:** New file: `evaluate.py`

Create a standalone evaluation script that produces a complete performance report for any saved model:

```python
"""
evaluate.py — Standalone model evaluation script.

Usage:
    python evaluate.py --model models/best_model --data data/test_EURUSD_1H_2023_2025.csv
"""
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from indicators import load_and_preprocess_data
from trading_env import ForexTradingEnv
from config import *
import pickle

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/best_model")
    parser.add_argument("--data",  default=TEST_DATA_PATH)
    args = parser.parse_args()

    df, feature_cols = load_and_preprocess_data(args.data)

    with open("models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    env = ForexTradingEnv(
        df=df,
        window_size=DEFAULT_WINDOW_SIZE,
        sl_options=DEFAULT_SL_OPTIONS,
        tp_options=DEFAULT_TP_OPTIONS,
        feature_columns=feature_cols,
        feature_mean=scaler.mean_.astype("float32"),
        feature_std=scaler.scale_.astype("float32"),
        spread_pips=DEFAULT_SPREAD_PIPS,
        max_slippage_pips=DEFAULT_SLIPPAGE_PIPS,
        random_start=False,
    )
    vec_env = DummyVecEnv([lambda: env])
    model   = PPO.load(args.model, env=vec_env)

    # Run episode
    obs = vec_env.reset()
    equity_curve = []
    closed_trades = []

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, dones, infos = vec_env.step(action)[:4]
        equity_curve.append(vec_env.get_attr("equity_usd")[0])
        t = vec_env.get_attr("last_trade_info")[0]
        if isinstance(t, dict) and t.get("event") == "CLOSE":
            closed_trades.append(t)
        if bool(dones[0]):
            break

    # Statistics
    equity  = np.array(equity_curve)
    returns = np.diff(equity) / (equity[:-1] + 1e-9)
    peak    = np.maximum.accumulate(equity)
    dd      = (peak - equity) / peak

    wins   = [t for t in closed_trades if t["net_pips"] > 0]
    losses = [t for t in closed_trades if t["net_pips"] <= 0]

    stats = {
        "final_equity":       round(float(equity[-1]), 2),
        "total_return_pct":   round((equity[-1] / 10000 - 1) * 100, 2),
        "sharpe_annualized":  round(float(returns.mean() / (returns.std() + 1e-9) * np.sqrt(6500)), 3),
        "max_drawdown_pct":   round(float(dd.max() * 100), 2),
        "calmar_ratio":       round(float((equity[-1]/10000 - 1) / (dd.max() + 1e-9)), 3),
        "num_trades":         len(closed_trades),
        "win_rate_pct":       round(len(wins) / len(closed_trades) * 100, 1) if closed_trades else 0,
        "avg_win_pips":       round(float(np.mean([t["net_pips"] for t in wins])), 2) if wins else 0,
        "avg_loss_pips":      round(float(np.mean([t["net_pips"] for t in losses])), 2) if losses else 0,
        "profit_factor":      round(abs(sum(t["net_pips"] for t in wins) / sum(t["net_pips"] for t in losses)), 3) if losses else 999,
        "expectancy_pips":    round(float(np.mean([t["net_pips"] for t in closed_trades])), 4) if closed_trades else 0,
    }

    print("\n=== Evaluation Report ===")
    for k, v in stats.items():
        print(f"  {k:<25} {v}")

    with open("logs/eval_report.json", "w") as f:
        json.dump(stats, f, indent=2)

    # Equity curve plot
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    axes[0].plot(equity_curve, label="Equity")
    axes[0].set_title("Equity Curve")
    axes[0].set_ylabel("USD")
    axes[0].legend()
    axes[1].fill_between(range(len(dd)), dd * 100, alpha=0.4, color="red", label="Drawdown %")
    axes[1].set_title("Drawdown")
    axes[1].set_ylabel("%")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig("logs/eval_equity_curve.png", dpi=150)
    plt.show()
    print("Saved: logs/eval_equity_curve.png, logs/eval_report.json")

if __name__ == "__main__":
    main()
```

---

## Phase 6 — Markov Chain Integration

**Goal:** Use a Markov chain as a separate probabilistic model that the RL agent can query to improve its entry and exit timing.

### Can Markov chains help here?

Yes — specifically for **regime detection** and **transition probability estimation**. A Markov chain cannot replace the RL agent (it has no memory of cumulative rewards and cannot optimize for long-term goals), but it can act as a **pre-filter** or **feature generator** that tells the agent the current market regime and the probability of transitioning to a different one.

### How it works

1. **Discretize price states** — partition the price action into a finite number of states (e.g., "trending up", "trending down", "ranging", "high volatility", "low volatility").
2. **Estimate a transition matrix** — from historical data, compute the probability of moving from state A to state B.
3. **Feed regime probabilities into the observation** — at each bar, compute the probability vector over all states and append it to the feature vector.
4. **Optionally gate entries** — only allow the RL agent to open a trade when the Markov chain indicates a favorable regime.

---

### Task 6.1 — Create `markov_chain.py`

**File:** New file: `markov_chain.py`

```python
"""
markov_chain.py

Implements a Hidden Markov Model (HMM) style Markov chain for EURUSD regime detection.

States:
  0: TREND_UP     — rising MA, positive returns, moderate ATR
  1: TREND_DOWN   — falling MA, negative returns, moderate ATR
  2: RANGE_HIGH_VOL — high ATR, mixed returns, tight MA spread
  3: RANGE_LOW_VOL  — low ATR, flat returns, flat MAs

The transition matrix is estimated from training data.
At inference time, the current state and next-state probabilities are returned.
"""

import numpy as np
import pandas as pd
import pickle
from typing import Tuple


def assign_state(row) -> int:
    """
    Assigns a market state to a single bar based on indicator values.
    This is the 'observation model' — maps continuous indicators to discrete states.

    The thresholds below are calibrated for EURUSD 1H. Adjust if using other instruments.
    """
    atr_ratio    = row.get("atr_ratio", 1.0)   # ATR relative to its 50-bar mean
    ma_spread    = row.get("ma_spread", 0.0)    # MA20 - MA50
    log_ret_6    = row.get("log_ret_6", 0.0)    # 6-bar log return
    rsi          = row.get("rsi_14", 50.0)

    is_high_vol  = atr_ratio > 1.2
    is_trending  = abs(ma_spread) > 0.0003  # Approximately 3 pips of MA separation

    if is_trending and log_ret_6 > 0 and rsi > 50:
        return 0  # TREND_UP
    elif is_trending and log_ret_6 < 0 and rsi < 50:
        return 1  # TREND_DOWN
    elif is_high_vol:
        return 2  # RANGE_HIGH_VOL
    else:
        return 3  # RANGE_LOW_VOL


STATE_LABELS = ["trend_up", "trend_down", "range_high_vol", "range_low_vol"]
N_STATES = len(STATE_LABELS)


def build_transition_matrix(df: pd.DataFrame) -> np.ndarray:
    """
    Estimates an (N_STATES x N_STATES) transition matrix from a training DataFrame.
    df must contain the columns expected by assign_state().

    Returns a row-stochastic matrix: T[i, j] = P(next state = j | current state = i)
    """
    states = df.apply(assign_state, axis=1).values
    counts = np.zeros((N_STATES, N_STATES), dtype=float)

    for t in range(len(states) - 1):
        counts[states[t], states[t + 1]] += 1

    # Add Laplace smoothing to avoid zero probabilities
    counts += 1.0

    # Normalize each row to sum to 1
    row_sums = counts.sum(axis=1, keepdims=True)
    transition_matrix = counts / row_sums

    return transition_matrix


def save_markov_model(transition_matrix: np.ndarray, path: str):
    with open(path, "wb") as f:
        pickle.dump({"transition_matrix": transition_matrix, "state_labels": STATE_LABELS}, f)


def load_markov_model(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


class MarkovRegimeFilter:
    """
    Stateful Markov chain regime tracker.
    Maintains a belief distribution over states (soft state) using the transition matrix.

    Usage:
        filter = MarkovRegimeFilter(transition_matrix)
        filter.reset()
        for each bar:
            probs = filter.update(current_state)
            # probs is a length-N_STATES array of current state probabilities
    """

    def __init__(self, transition_matrix: np.ndarray):
        self.T = transition_matrix
        self.belief = np.ones(N_STATES) / N_STATES  # Uniform prior

    def reset(self):
        self.belief = np.ones(N_STATES) / N_STATES

    def update(self, observed_state: int) -> np.ndarray:
        """
        Updates belief given the observed state and returns the posterior.
        Uses a simple Bayesian update: multiply by likelihood (1 if matches, 0.1 otherwise),
        then predict next step using transition matrix.
        """
        # Likelihood: sharp update toward observed state
        likelihood = np.full(N_STATES, 0.05)
        likelihood[observed_state] = 1.0

        # Update
        posterior = self.belief * likelihood
        posterior /= posterior.sum()

        # Predict next state
        self.belief = posterior @ self.T

        return posterior.astype(np.float32)

    def next_state_probabilities(self) -> np.ndarray:
        """Returns the predicted next-state probability distribution."""
        return (self.belief @ self.T).astype(np.float32)
```

---

### Task 6.2 — Train the Markov chain from training data

**File:** `train_agent.py`

Add to the training script after data loading:

```python
from markov_chain import build_transition_matrix, save_markov_model
import os

# Build Markov transition matrix from training data only
print("Building Markov transition matrix...")
transition_matrix = build_transition_matrix(train_df)
os.makedirs("models", exist_ok=True)
save_markov_model(transition_matrix, "models/markov_model.pkl")
print("Transition matrix:")
print(np.round(transition_matrix, 3))
```

---

### Task 6.3 — Integrate Markov state into the observation space

**File:** `trading_env.py`

This is the recommended integration path — add the 4-state probability vector directly to the observation so the RL agent can condition on it.

Add to `ForexTradingEnv.__init__`:

```python
from markov_chain import MarkovRegimeFilter, assign_state, N_STATES, load_markov_model
import os

# Load Markov model if available
markov_path = "models/markov_model.pkl"
if os.path.exists(markov_path):
    markov_data = load_markov_model(markov_path)
    self.markov_filter = MarkovRegimeFilter(markov_data["transition_matrix"])
    self.use_markov = True
else:
    self.markov_filter = None
    self.use_markov = False

self.markov_num_features = N_STATES if self.use_markov else 0
```

Update `state_num_features`:
```python
# Before: 3 state features (position, time_in_trade, unrealized_pips)
# After: 3 + N_STATES = 7 state features
self.state_num_features = 3 + self.markov_num_features
self.num_features = self.base_num_features + self.state_num_features
```

Update `_get_state_features`:
```python
def _get_state_features(self):
    pos         = float(self.position)
    t_norm      = float(self.time_in_trade) / 1000.0
    unreal_pips = float(self._compute_unrealized_pips()) if self.position != 0 else 0.0
    unreal_sc   = unreal_pips / 100.0

    base = np.array([pos, t_norm, unreal_sc], dtype=np.float32)

    if self.use_markov and self.markov_filter is not None:
        row = self.df.iloc[self.current_step]
        current_state = assign_state(row)
        markov_probs = self.markov_filter.update(current_state)
        base = np.concatenate([base, markov_probs])

    return base
```

Update `reset` to reset the Markov filter:
```python
def reset(self, seed=None, options=None):
    super().reset(seed=seed)
    self._reset_state()
    if self.use_markov and self.markov_filter is not None:
        self.markov_filter.reset()
    ...
```

---

### Task 6.4 — Optional: Markov entry gate

As an alternative or complement to using Markov as features, implement an entry gate that blocks the RL agent from opening trades when the Markov chain predicts high-volatility range conditions (state 2) with high probability:

```python
# In trading_env.py, step(), OPEN action block:
elif act_type == "OPEN":
    if self.position == 0:
        # Markov gate: block trades in unfavorable regimes
        if self.use_markov:
            next_probs = self.markov_filter.next_state_probabilities()
            high_vol_prob = next_probs[2]  # P(next state = RANGE_HIGH_VOL)
            if high_vol_prob > 0.6:
                # Override OPEN to HOLD — add small penalty to discourage wasted action
                reward_pips -= 0.05
                act_type = "HOLD"  # Block the trade
        if act_type == "OPEN":  # Only if not blocked
            self._open_position(direction=direction, sl_pips=sl_pips, tp_pips=tp_pips)
```

**Important:** Only enable the Markov gate after the base RL training is stable. It should be a configurable flag: `use_markov_gate: bool = False` by default.

---

## Phase 7 — Risk Management Layer

**Goal:** Protect capital at the account level, independent of what the RL agent decides.

This layer sits between the model's output and the actual order execution. Even if the model performs perfectly in backtesting, real markets have regime changes. The risk layer provides a hard safety net.

---

### Task 7.1 — Create `risk_manager.py`

**File:** New file: `risk_manager.py`

```python
"""
risk_manager.py

Account-level risk management rules applied BEFORE any trade is executed.
All rules are stateless checks that take current account state and return
a boolean (True = allow, False = block).

These rules operate independently of the RL agent. The agent proposes a trade;
the risk manager approves or rejects it.
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class AccountState:
    equity_usd:          float
    peak_equity_usd:     float
    open_position:       int        # 0 = flat, 1 = long, -1 = short
    daily_pnl_usd:       float      # PnL since last daily reset
    trades_today:        int
    consecutive_losses:  int
    current_drawdown_pct: float     # (peak - equity) / peak


@dataclass
class RiskConfig:
    max_daily_drawdown_pct:     float = 0.05   # Halt if daily loss exceeds 5% of equity
    max_total_drawdown_pct:     float = 0.15   # Halt if total drawdown from peak exceeds 15%
    max_trades_per_day:         int   = 10     # No more than 10 trades per 24h
    max_consecutive_losses:     int   = 5      # Pause after 5 losses in a row
    min_equity_usd:             float = 5000.0 # Hard stop if equity falls below this
    max_position_pct_of_equity: float = 0.02   # Max 2% risk per trade (not yet implemented)


class RiskManager:

    def __init__(self, config: RiskConfig = None):
        self.config    = config or RiskConfig()
        self.is_halted = False
        self.halt_reason: Optional[str] = None

    def check(self, state: AccountState) -> tuple[bool, str]:
        """
        Returns (allowed: bool, reason: str).
        'allowed' is False if any risk rule is violated.
        """
        c = self.config

        if self.is_halted:
            return False, f"System halted: {self.halt_reason}"

        if state.equity_usd < c.min_equity_usd:
            self._halt("Equity below minimum threshold")
            return False, self.halt_reason

        if state.current_drawdown_pct >= c.max_total_drawdown_pct:
            self._halt(f"Total drawdown {state.current_drawdown_pct:.1%} exceeds max {c.max_total_drawdown_pct:.1%}")
            return False, self.halt_reason

        daily_dd = abs(state.daily_pnl_usd) / (state.equity_usd + 1e-9)
        if state.daily_pnl_usd < 0 and daily_dd >= c.max_daily_drawdown_pct:
            return False, f"Daily drawdown limit reached ({daily_dd:.1%})"

        if state.trades_today >= c.max_trades_per_day:
            return False, f"Max daily trades ({c.max_trades_per_day}) reached"

        if state.consecutive_losses >= c.max_consecutive_losses:
            return False, f"Consecutive loss limit ({c.max_consecutive_losses}) reached — cooling off"

        return True, "OK"

    def _halt(self, reason: str):
        self.is_halted = True
        self.halt_reason = reason
        print(f"[RISK MANAGER] HALT: {reason}")

    def reset_halt(self):
        """Manual override to resume after investigating a halt."""
        self.is_halted = False
        self.halt_reason = None
        print("[RISK MANAGER] Halt cleared. Resuming.")

    def reset_daily(self):
        """Call this at the start of each trading day."""
        pass  # Daily PnL tracking is stateful in the caller
```

---

### Task 7.2 — Add position sizing

**File:** `risk_manager.py` (add to existing file)

The current system always trades exactly 1 lot. This is not capital-efficient and is dangerous at scale.

```python
def compute_lot_size(
    equity_usd:    float,
    risk_pct:      float,   # e.g. 0.01 = risk 1% of equity per trade
    sl_pips:       float,   # stop-loss in pips
    pip_value_usd: float = 10.0,  # for 1 lot EURUSD at standard account
    min_lots:      float = 0.01,
    max_lots:      float = 2.0,
) -> float:
    """
    Calculates lot size using fixed fractional position sizing.

    Formula: lot_size = (equity * risk_pct) / (sl_pips * pip_value_per_lot)
    """
    if sl_pips <= 0 or pip_value_usd <= 0:
        return min_lots

    risk_amount = equity_usd * risk_pct
    lot_size    = risk_amount / (sl_pips * pip_value_usd)
    lot_size    = round(lot_size, 2)  # Round to nearest 0.01 lot
    lot_size    = max(min_lots, min(max_lots, lot_size))

    return lot_size
```

---

## Phase 8 — Live Trading Infrastructure

**Goal:** Connect the trained model to real (or paper) trading via a broker API. This phase is NOT to be implemented until the model has demonstrated ≥ 6 months of out-of-sample profitability.

---

### Task 8.1 — Design the execution interface

**File:** New file: `broker/base_broker.py`

Use an abstract base class so the system can be connected to any broker without changing the agent code:

```python
"""
broker/base_broker.py

Abstract broker interface. Implement this for any specific broker (OANDA, MetaTrader, Interactive Brokers, etc.)
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class Order:
    symbol:     str
    direction:  int    # 1 = long, -1 = short
    lot_size:   float
    sl_price:   float
    tp_price:   float
    order_type: str = "MARKET"


@dataclass
class Position:
    symbol:        str
    direction:     int
    lot_size:      float
    entry_price:   float
    sl_price:      float
    tp_price:      float
    unrealized_pnl: float = 0.0


class BaseBroker(ABC):

    @abstractmethod
    def get_current_price(self, symbol: str) -> tuple[float, float]:
        """Returns (bid, ask)."""
        pass

    @abstractmethod
    def get_account_equity(self) -> float:
        """Returns current account equity in USD."""
        pass

    @abstractmethod
    def submit_order(self, order: Order) -> str:
        """Submits an order. Returns an order ID."""
        pass

    @abstractmethod
    def close_position(self, symbol: str) -> float:
        """Closes all open positions on symbol. Returns realized PnL."""
        pass

    @abstractmethod
    def get_open_position(self, symbol: str) -> Optional[Position]:
        """Returns current open position or None if flat."""
        pass

    @abstractmethod
    def get_ohlcv(self, symbol: str, timeframe: str, count: int) -> "pd.DataFrame":
        """Returns recent OHLCV data as a DataFrame."""
        pass
```

---

### Task 8.2 — Implement OANDA broker adapter

**File:** New file: `broker/oanda_broker.py`

OANDA offers a REST API with a Python SDK (`oandapyV20`). This is the recommended starting broker for retail forex algo trading.

```python
"""
broker/oanda_broker.py

OANDA v20 REST API broker adapter.
Requires: pip install oandapyV20
Credentials: Set OANDA_ACCOUNT_ID and OANDA_API_KEY in environment variables.
Use environment="practice" for paper trading, "live" for real money.
"""
import os
import pandas as pd
import oandapyV20
import oandapyV20.endpoints.orders     as orders
import oandapyV20.endpoints.trades     as trades
import oandapyV20.endpoints.pricing    as pricing
import oandapyV20.endpoints.instruments as instruments
import oandapyV20.endpoints.accounts   as accounts_ep
from broker.base_broker import BaseBroker, Order, Position
from typing import Optional


class OANDABroker(BaseBroker):

    def __init__(self, environment: str = "practice"):
        self.account_id = os.environ["OANDA_ACCOUNT_ID"]
        api_key         = os.environ["OANDA_API_KEY"]
        self.client     = oandapyV20.API(access_token=api_key, environment=environment)

    def get_current_price(self, symbol: str) -> tuple[float, float]:
        r = pricing.PricingInfo(self.account_id, params={"instruments": symbol})
        self.client.request(r)
        price = r.response["prices"][0]
        bid   = float(price["bids"][0]["price"])
        ask   = float(price["asks"][0]["price"])
        return bid, ask

    def get_account_equity(self) -> float:
        r = accounts_ep.AccountSummary(self.account_id)
        self.client.request(r)
        return float(r.response["account"]["NAV"])

    def submit_order(self, order: Order) -> str:
        direction = "BUY" if order.direction == 1 else "SELL"
        units     = int(order.lot_size * 100000)
        if order.direction == -1:
            units = -units

        data = {
            "order": {
                "type": "MARKET",
                "instrument": order.symbol,
                "units": str(units),
                "stopLossOnFill":   {"price": f"{order.sl_price:.5f}"},
                "takeProfitOnFill": {"price": f"{order.tp_price:.5f}"},
            }
        }
        r = orders.OrderCreate(self.account_id, data=data)
        self.client.request(r)
        return r.response["orderFillTransaction"]["id"]

    def close_position(self, symbol: str) -> float:
        r = trades.OpenTrades(self.account_id)
        self.client.request(r)
        for trade in r.response.get("trades", []):
            if trade["instrument"] == symbol:
                close_r = trades.TradeClose(self.account_id, trade["id"])
                self.client.request(close_r)
                return float(close_r.response["orderFillTransaction"]["pl"])
        return 0.0

    def get_open_position(self, symbol: str) -> Optional[Position]:
        r = trades.OpenTrades(self.account_id)
        self.client.request(r)
        for trade in r.response.get("trades", []):
            if trade["instrument"] == symbol:
                units = int(trade["currentUnits"])
                return Position(
                    symbol=symbol,
                    direction=1 if units > 0 else -1,
                    lot_size=abs(units) / 100000,
                    entry_price=float(trade["price"]),
                    sl_price=float(trade.get("stopLossOrder", {}).get("price", 0)),
                    tp_price=float(trade.get("takeProfitOrder", {}).get("price", 0)),
                    unrealized_pnl=float(trade["unrealizedPL"]),
                )
        return None

    def get_ohlcv(self, symbol: str, timeframe: str = "H1", count: int = 200) -> pd.DataFrame:
        params = {"count": count, "granularity": timeframe}
        r = instruments.InstrumentsCandles(symbol, params=params)
        self.client.request(r)
        candles = r.response["candles"]
        rows = []
        for c in candles:
            if not c["complete"]:
                continue
            rows.append({
                "Time (EET)": pd.to_datetime(c["time"]),
                "Open":    float(c["mid"]["o"]),
                "High":    float(c["mid"]["h"]),
                "Low":     float(c["mid"]["l"]),
                "Close":   float(c["mid"]["c"]),
                "Volume":  int(c["volume"]),
            })
        df = pd.DataFrame(rows).set_index("Time (EET)")
        return df
```

---

### Task 8.3 — Create the live trading loop

**File:** New file: `live_trader.py`

```python
"""
live_trader.py

Main live trading loop. Runs once per hour (on the H1 close).
Fetches new data, runs feature engineering, gets model action, applies risk check, executes.

Usage:
    python live_trader.py --model models/best_model --symbol EUR_USD --env practice
"""
import time, argparse, pickle, logging
from datetime import datetime, timezone

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from broker.oanda_broker import OANDABroker
from risk_manager import RiskManager, RiskConfig, AccountState, compute_lot_size
from indicators import load_and_preprocess_data
from trading_env import ForexTradingEnv
from config import *

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  default="models/best_model")
    parser.add_argument("--symbol", default="EUR_USD")
    parser.add_argument("--env",    default="practice")
    args = parser.parse_args()

    # Load model and scaler
    with open("models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    broker       = OANDABroker(environment=args.env)
    risk_manager = RiskManager(RiskConfig())

    # We need a dummy env to load the model — will be replaced on each step
    # Load model without env first, then set env
    model = PPO.load(args.model)

    peak_equity  = broker.get_account_equity()
    daily_pnl    = 0.0
    trades_today = 0
    consec_losses = 0
    last_day     = datetime.now(timezone.utc).date()

    log.info("Live trader started. Symbol: %s, Env: %s", args.symbol, args.env)

    while True:
        now = datetime.now(timezone.utc)

        # Reset daily counters
        if now.date() != last_day:
            daily_pnl    = 0.0
            trades_today = 0
            last_day     = now.date()
            log.info("Daily counters reset.")

        # Fetch latest OHLCV data
        try:
            raw_df = broker.get_ohlcv(args.symbol, timeframe="H1", count=300)
        except Exception as e:
            log.error("Failed to fetch data: %s", e)
            time.sleep(60)
            continue

        # Feature engineering
        try:
            # indicators.py expects a CSV path, so we fake it with the df directly
            # Refactor load_and_preprocess_data to accept either a path OR a DataFrame
            from indicators import compute_indicators_from_df
            df, feature_cols = compute_indicators_from_df(raw_df)
        except Exception as e:
            log.error("Feature engineering failed: %s", e)
            time.sleep(60)
            continue

        # Normalize
        feature_mean = scaler.mean_.astype(np.float32)
        feature_std  = scaler.scale_.astype(np.float32)

        # Build single-step environment for inference
        live_env = ForexTradingEnv(
            df=df,
            window_size=DEFAULT_WINDOW_SIZE,
            sl_options=DEFAULT_SL_OPTIONS,
            tp_options=DEFAULT_TP_OPTIONS,
            feature_columns=feature_cols,
            feature_mean=feature_mean,
            feature_std=feature_std,
            random_start=False,
        )
        live_env.current_step = len(df) - 1  # Point to the latest bar

        # Get current position from broker
        open_pos = broker.get_open_position(args.symbol)
        live_env.position = open_pos.direction if open_pos else 0

        obs    = live_env._get_observation()
        equity = broker.get_account_equity()

        # Update peak
        if equity > peak_equity:
            peak_equity = equity

        current_dd = (peak_equity - equity) / peak_equity

        account_state = AccountState(
            equity_usd=equity,
            peak_equity_usd=peak_equity,
            open_position=live_env.position,
            daily_pnl_usd=daily_pnl,
            trades_today=trades_today,
            consecutive_losses=consec_losses,
            current_drawdown_pct=current_dd,
        )

        # Risk check
        allowed, reason = risk_manager.check(account_state)
        if not allowed:
            log.warning("Trade blocked by risk manager: %s", reason)
            time.sleep(3600)
            continue

        # Model inference
        action, _ = model.predict(obs[np.newaxis], deterministic=True)
        act_type, direction, sl_pips, tp_pips = live_env.action_map[int(action[0])]
        log.info("Model action: %s | dir=%s | sl=%s | tp=%s", act_type, direction, sl_pips, tp_pips)

        # Execute
        bid, ask = broker.get_current_price(args.symbol)
        mid_price = (bid + ask) / 2

        if act_type == "OPEN" and live_env.position == 0:
            lot_size = compute_lot_size(equity, risk_pct=0.01, sl_pips=sl_pips)
            pip_v    = 0.0001
            if direction == 1:  # long
                sl_price = ask - sl_pips * pip_v
                tp_price = ask + tp_pips * pip_v
            else:               # short
                sl_price = bid + sl_pips * pip_v
                tp_price = bid - tp_pips * pip_v

            from broker.base_broker import Order
            order = Order(args.symbol, direction, lot_size, sl_price, tp_price)
            order_id = broker.submit_order(order)
            trades_today += 1
            log.info("Order submitted: %s, lot_size=%.2f, sl=%.5f, tp=%.5f", order_id, lot_size, sl_price, tp_price)

        elif act_type == "CLOSE" and live_env.position != 0:
            pnl = broker.close_position(args.symbol)
            daily_pnl += pnl
            if pnl < 0:
                consec_losses += 1
            else:
                consec_losses = 0
            log.info("Position closed. PnL: %.2f USD", pnl)

        # Wait for next bar close (sleep until next hour)
        sleep_seconds = 3600 - (now.minute * 60 + now.second)
        log.info("Sleeping %d seconds until next bar.", sleep_seconds)
        time.sleep(sleep_seconds)


if __name__ == "__main__":
    main()
```

---

### Task 8.4 — Refactor `indicators.py` to accept a DataFrame directly

**File:** `indicators.py`

The live trader needs to compute indicators on data fetched from the broker, not a CSV. Add an overloaded entry point:

```python
def compute_indicators_from_df(df: pd.DataFrame):
    """
    Same as load_and_preprocess_data but accepts a DataFrame instead of a file path.
    The DataFrame must have columns: Open, High, Low, Close, Volume.
    """
    # Strip spaces from column names
    df = df.copy()
    df.columns = df.columns.str.strip()

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Apply the same indicator logic as load_and_preprocess_data
    # (extract the indicator computation block to a private helper to avoid duplication)
    return _compute_all_indicators(df)


def _compute_all_indicators(df: pd.DataFrame):
    """Internal helper — computes all indicators and returns (df, feature_cols)."""
    # ... (move indicator computation from load_and_preprocess_data here)
    pass
```

Refactor `load_and_preprocess_data` to call `_compute_all_indicators` after loading the CSV.

---

## Phase 9 — Monitoring & Observability

**Goal:** Know what the bot is doing in real-time and catch problems before they become expensive.

---

### Task 9.1 — Add structured logging throughout the system

**File:** All files

Every significant event should be logged with a structured format that can be parsed by log aggregators (Datadog, Grafana Loki, etc.):

```python
import logging
import json

class JSONFormatter(logging.Formatter):
    def format(self, record):
        return json.dumps({
            "timestamp": self.formatTime(record),
            "level":     record.levelname,
            "message":   record.getMessage(),
            "module":    record.module,
        })

handler = logging.FileHandler("logs/live_trader.jsonl")
handler.setFormatter(JSONFormatter())
log.addHandler(handler)
```

---

### Task 9.2 — Add a health check endpoint

**File:** New file: `health_server.py`

```python
"""
health_server.py

Minimal HTTP server that exposes system status for external monitoring.
Run alongside live_trader.py in a separate process or thread.

Endpoints:
  GET /health  — returns 200 if system is running, 503 if halted
  GET /stats   — returns current equity, PnL, drawdown as JSON
"""
from http.server import BaseHTTPRequestHandler, HTTPServer
import json, threading

_state = {
    "equity": 0.0,
    "daily_pnl": 0.0,
    "drawdown_pct": 0.0,
    "is_halted": False,
    "last_action": "none",
    "last_update": "",
}

def update_state(**kwargs):
    _state.update(kwargs)

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            code = 503 if _state["is_halted"] else 200
            self.send_response(code)
            self.end_headers()
        elif self.path == "/stats":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(_state).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass  # Suppress access logs

def start(port: int = 8080):
    server = HTTPServer(("0.0.0.0", port), Handler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    print(f"Health server running on :{port}")
```

---

### Task 9.3 — Add TensorBoard training metrics

**File:** `train_agent.py`

SB3 already supports TensorBoard via `tensorboard_log`. Add custom scalars for tracking trading-specific metrics during training using a custom callback:

```python
from stable_baselines3.common.callbacks import BaseCallback

class TradingMetricsCallback(BaseCallback):
    """Logs trading-specific metrics to TensorBoard every eval_freq steps."""

    def __init__(self, eval_env, eval_freq=50_000, verbose=0):
        super().__init__(verbose)
        self.eval_env  = eval_env
        self.eval_freq = eval_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq != 0:
            return True

        obs = self.eval_env.reset()
        equity_curve, trades = [], []

        while True:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, _, dones, infos = self.eval_env.step(action)[:4]
            equity_curve.append(self.eval_env.get_attr("equity_usd")[0])
            t = self.eval_env.get_attr("last_trade_info")[0]
            if isinstance(t, dict) and t.get("event") == "CLOSE":
                trades.append(t)
            if bool(dones[0]):
                break

        import numpy as np
        equity  = np.array(equity_curve)
        returns = np.diff(equity) / (equity[:-1] + 1e-9)
        sharpe  = returns.mean() / (returns.std() + 1e-9) * np.sqrt(6500)

        self.logger.record("eval/final_equity",  float(equity[-1]))
        self.logger.record("eval/sharpe",        float(sharpe))
        self.logger.record("eval/num_trades",    len(trades))

        if trades:
            wins = [t for t in trades if t["net_pips"] > 0]
            self.logger.record("eval/win_rate", len(wins) / len(trades))

        return True
```

---

## Nice-to-Haves & Future Enhancements

These are not required for capital readiness but significantly improve the system over time.

### Multi-pair trading
- Train separate models for GBPUSD, USDJPY, AUDUSD.
- Add a portfolio-level risk manager that limits total open exposure across all pairs.
- Correlate signals — avoid being long EURUSD and GBPUSD simultaneously (highly correlated).

### Multi-timeframe observation
- Fetch 15-minute and 4-hour bars as auxiliary features.
- Use a hierarchical observation: H1 window (30 bars) + H4 context (10 bars) + M15 detail (20 bars).
- This requires a custom observation space — three separate arrays flattened and concatenated.

### LSTM / Transformer policy
- Replace the default `MlpPolicy` with an `LstmPolicy` (available in SB3-contrib's `RecurrentPPO`).
- An LSTM naturally handles the temporal structure of price data without requiring a manual rolling window.
- This is a significant architectural change — treat as a separate experimental branch.

### Regime-adaptive sizing
- Use the Markov chain regime probabilities to scale position size.
- In a high-confidence trend regime: allow up to 1.5x normal lot size.
- In a range/high-volatility regime: reduce to 0.5x or skip entirely.

### Automatic retraining pipeline
- Set up a weekly retraining job that:
  1. Downloads the latest 4 weeks of H1 data from OANDA.
  2. Re-trains the model on an expanding window.
  3. Evaluates on the most recent 2 weeks.
  4. If Sharpe improves, promotes the new model to production. If not, keeps the current.
- This prevents the model from becoming stale as market microstructure evolves.

### News/event awareness
- Integrate an economic calendar API (e.g., Investing.com, ForexFactory API).
- Block new trades in the 30 minutes before and after high-impact news events (NFP, CPI, FOMC).
- This alone can dramatically reduce drawdown caused by model confusion during news volatility.

### Shadow mode / paper trading validation
- Before going live, run the model in "shadow mode" — live data, real signals, but no actual orders.
- Log what it would have done and compare against actual market outcomes.
- Run shadow mode for at least 3 months before real capital.

### A/B model testing
- Run two model versions simultaneously, each with a small allocation.
- Compare performance over rolling 30-day windows.
- Gradually shift allocation to the better-performing model.

---

## File Structure (Target)

```
Reinforcement-Trading/
├── config.py                    # Central config — paths, constants, default params
├── indicators.py                # Feature engineering — all technical indicators
├── trading_env.py               # Gymnasium environment — core RL loop
├── markov_chain.py              # Markov chain regime detection
├── risk_manager.py              # Account-level risk rules and position sizing
├── train_agent.py               # Training script — data loading, PPO, callbacks
├── evaluate.py                  # Standalone evaluation — stats, plots, report
├── live_trader.py               # Live trading loop — broker integration
├── health_server.py             # HTTP health/status endpoint
│
├── broker/
│   ├── __init__.py
│   ├── base_broker.py           # Abstract broker interface
│   └── oanda_broker.py          # OANDA v20 REST API adapter
│
├── data/
│   ├── train_EURUSD_1H_2020_2023.csv
│   └── test_EURUSD_1H_2023_2025.csv
│
├── models/
│   ├── best_model.zip           # Best PPO model (SB3 format)
│   ├── scaler.pkl               # Fitted StandardScaler (match train data)
│   └── markov_model.pkl         # Fitted Markov transition matrix
│
├── checkpoints/
│   └── ppo_eurusd_*.zip         # Periodic training checkpoints
│
├── logs/
│   ├── eval_report.json         # Latest evaluation statistics
│   ├── eval_equity_curve.png    # Equity curve chart
│   ├── trade_history.csv        # All closed trades log
│   └── live_trader.jsonl        # Structured live trading log
│
├── tensorboard_log/             # TensorBoard training metrics
│
├── tests/
│   ├── test_env.py              # Unit tests for trading environment
│   ├── test_indicators.py       # Unit tests for feature engineering
│   └── test_risk_manager.py     # Unit tests for risk rules
│
├── requirements.txt
├── README.md
└── IMPROVEMENT_PLAN.md          # This file
```

---

## Definition of Done

The system is considered capital-ready when ALL of the following are true:

| Criterion | Target | How to Measure |
|---|---|---|
| Sharpe Ratio (OOS) | ≥ 1.5 | `evaluate.py` on test dataset |
| Max Drawdown (OOS) | ≤ 15% | `evaluate.py` on test dataset |
| Win Rate | ≥ 40% | `evaluate.py` trade stats |
| Profit Factor | ≥ 1.3 | `evaluate.py` trade stats |
| Walk-Forward passes | 3 of 4 windows positive Sharpe | `train_agent.py` walk-forward eval |
| Paper trading period | ≥ 90 days | Shadow mode log |
| Risk manager tested | All 5 rules trigger correctly | `tests/test_risk_manager.py` |
| Markov chain accuracy | State classification accuracy ≥ 60% on test data | Manual evaluation |
| Live data pipeline tested | 48h continuous run without crash | Health endpoint logs |
| Retraining pipeline exists | Model auto-updates weekly | Cron job + eval report |

---

*Last updated: Phase 1 through Phase 9 + Nice-to-Haves. This document should be updated by any contributor completing a phase.*
