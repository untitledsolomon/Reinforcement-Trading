import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from indicators import load_and_preprocess_data
from trading_env import ForexTradingEnv
from config import TEST_DATA_PATH, MODEL_DIR, DEFAULT_WINDOW_SIZE, DEFAULT_SL_OPTIONS, DEFAULT_TP_OPTIONS, DEFAULT_SPREAD_PIPS, DEFAULT_SLIPPAGE_PIPS
import os
import pickle


def run_one_episode(model, vec_env, deterministic=True):
    obs = vec_env.reset()
    equity_curve = []
    closed_trades = []

    while True:
        action, _ = model.predict(obs, deterministic=deterministic)
        step_out = vec_env.step(action)

        if len(step_out) == 4:
            obs, rewards, dones, infos = step_out
            done = bool(dones[0])
        else:
            obs, rewards, terminated, truncated, infos = step_out
            done = bool(terminated[0] or truncated[0])

        equity_curve.append(vec_env.get_attr("equity_usd")[0])

        trade_info = vec_env.get_attr("last_trade_info")[0]
        if isinstance(trade_info, dict) and trade_info.get("event") == "CLOSE":
            closed_trades.append(trade_info)

        if done:
            break

    return equity_curve, closed_trades


def compute_statistics(equity_curve: list, closed_trades: list, initial_equity: float = 10000.0):
    import numpy as np

    equity = np.array(equity_curve)
    returns = np.diff(equity) / (equity[:-1] + 1e-9)

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


def main():
    df, feature_cols = load_and_preprocess_data(TEST_DATA_PATH)
    test_df = df.copy()

    # Load the scaler that was saved during training
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
    if os.path.exists(scaler_path):
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        feature_mean = scaler.mean_.astype(np.float32)
        feature_std  = scaler.scale_.astype(np.float32)
    else:
        feature_mean = None
        feature_std  = None

    # Must match training params
    SL_OPTS = DEFAULT_SL_OPTIONS
    TP_OPTS = DEFAULT_TP_OPTIONS
    WIN = DEFAULT_WINDOW_SIZE

    test_env = ForexTradingEnv(
        df=test_df,
            window_size=WIN,
            sl_options=SL_OPTS,
            tp_options=TP_OPTS,
            spread_pips=DEFAULT_SPREAD_PIPS,
            commission_pips=0.0,
            max_slippage_pips=DEFAULT_SLIPPAGE_PIPS,
            random_start=False,
            episode_max_steps=None,
            feature_columns=feature_cols,
            feature_mean=feature_mean,
            feature_std=feature_std,
            hold_reward_weight=0.00, # Use 0.0 for test to see raw performance unless shaping is desired in eval
            open_penalty_pips=0.0,
            time_penalty_pips=0.0,
            unrealized_delta_weight=0.0
    )

    vec_test_env = DummyVecEnv([lambda: test_env])

    # Load best model
    best_model_path = os.path.join(MODEL_DIR, "best_model.zip")
    if not os.path.exists(best_model_path):
        # Fallback to current model if best_model doesn't exist yet
        best_model_path = os.path.join(MODEL_DIR, "best_model")

    if os.path.exists(best_model_path) or os.path.exists(best_model_path + ".zip"):
        model = PPO.load(best_model_path, env=vec_test_env)
    else:
        print(f"Warning: Model not found at {best_model_path}. Exiting.")
        return

    equity_curve, closed_trades = run_one_episode(model, vec_test_env, deterministic=True)

    # Compute and print statistics
    stats = compute_statistics(equity_curve, closed_trades)
    print("\n=== Test Statistics ===")
    for k, v in stats.items():
        print(f"{k}: {v}")

    # Save statistics to logs
    os.makedirs("logs", exist_ok=True)
    import json
    with open("logs/test_stats.json", "w") as f:
        json.dump(stats, f, indent=4)

    # Save trades
    if closed_trades:
        trades_df = pd.DataFrame(closed_trades)
        out_csv = "trade_history_output.csv"
        trades_df.to_csv(out_csv, index=False)
        print(f"Closed trade history saved to {out_csv}")
    else:
        print("No closed trades recorded.")

    # Plot equity
    plt.figure(figsize=(10, 6))
    plt.plot(equity_curve, label="Equity (Test)")
    plt.title("Equity Curve - Evaluation")
    plt.xlabel("Steps")
    plt.ylabel("Equity ($)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("logs/test_equity_curve.png")
    plt.show()


if __name__ == "__main__":
    main()
