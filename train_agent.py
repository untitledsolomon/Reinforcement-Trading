import os
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

from indicators import load_and_preprocess_data, fit_scaler
from trading_env import ForexTradingEnv
import pickle
from config import TRAIN_DATA_PATH, MODEL_DIR, CHECKPOINT_DIR, DEFAULT_WINDOW_SIZE, DEFAULT_SL_OPTIONS, DEFAULT_TP_OPTIONS, DEFAULT_SPREAD_PIPS, DEFAULT_SLIPPAGE_PIPS


def evaluate_model(model: PPO, eval_env: DummyVecEnv, deterministic: bool = True):
    obs = eval_env.reset()
    equity_curve = []

    while True:
        action, _ = model.predict(obs, deterministic=deterministic)
        step_out = eval_env.step(action)

        if len(step_out) == 4:
            obs, rewards, dones, infos = step_out
            done = bool(dones[0])
        else:
            obs, rewards, terminated, truncated, infos = step_out
            done = bool(terminated[0] or truncated[0])

        info = infos[0] if isinstance(infos, (list, tuple)) else infos
        # use equity from info (state *before* DummyVecEnv reset)
        eq = info.get("equity_usd", eval_env.get_attr("equity_usd")[0])
        equity_curve.append(eq)

        if done:
            break

    final_equity = float(equity_curve[-1])
    return equity_curve, final_equity



def main():
    df, feature_cols = load_and_preprocess_data(TRAIN_DATA_PATH)

    # Time split 80/20
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].copy()
    val_df   = df.iloc[split_idx:].copy()

    # Fit ONLY on training data
    scaler = fit_scaler(train_df, feature_cols)
    feature_mean = scaler.mean_.astype(np.float32)
    feature_std  = scaler.scale_.astype(np.float32)

    # Save scaler
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(os.path.join(MODEL_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    print("Training bars:", len(train_df))
    print("Testing bars :", len(val_df))

    # ---- Env factories ----
    SL_OPTS = DEFAULT_SL_OPTIONS
    TP_OPTS = DEFAULT_TP_OPTIONS
    WIN = DEFAULT_WINDOW_SIZE

    # Train env: random starts to reduce memorization
    def make_train_env():
        return ForexTradingEnv(
            df=train_df,
            window_size=WIN,
            sl_options=SL_OPTS,
            tp_options=TP_OPTS,
            spread_pips=DEFAULT_SPREAD_PIPS,
            commission_pips=0.0,
            max_slippage_pips=DEFAULT_SLIPPAGE_PIPS,
            random_start=True,
            min_episode_steps=1000,
            episode_max_steps=2000,
            feature_columns=feature_cols,
            feature_mean=feature_mean,
            feature_std=feature_std,
            hold_reward_weight=0.0,#0.05
            open_penalty_pips=0.0,      # 0.5 half a pip per open
            time_penalty_pips=0.0,     # 0.02 pips per bar in trade
            unrealized_delta_weight=0.0
        )

    # Train-eval env: deterministic start, NO random starts (so curve is stable/reproducible)
    def make_train_eval_env():
        return ForexTradingEnv(
            df=train_df,
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
            hold_reward_weight=0.00,
            open_penalty_pips=0.0,      # half a pip per open
            time_penalty_pips=0.0,     # 0.02 pips per bar in trade
            unrealized_delta_weight=0.0
        )

    # Test-eval env: deterministic
    def make_test_eval_env():
        return ForexTradingEnv(
            df=val_df,
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
            hold_reward_weight=0.00,
            open_penalty_pips=0.0,      # half a pip per open
            time_penalty_pips=0.00,     # 0.02 pips per bar in trade
            unrealized_delta_weight=0.0
        )

    train_vec_env = DummyVecEnv([make_train_env])
    train_eval_env = DummyVecEnv([make_train_eval_env])
    test_eval_env = DummyVecEnv([make_test_eval_env])

    # ---- Model ----
    model = PPO(
        policy="MlpPolicy",
        env=train_vec_env,
        verbose=1,
        tensorboard_log="./tensorboard_log/"
    )

    # ---- Checkpoints ----
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path=CHECKPOINT_DIR,
        name_prefix="ppo_eurusd"
    )

    # ---- Train ----
    total_timesteps = 600000
    model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)

    # ---- Select best model by OOS final equity ----
    equity_curve_test_last, final_equity_test_last = evaluate_model(model, test_eval_env)
    print(f"[OOS Eval] Last model final equity: {final_equity_test_last:.2f}")

    best_equity = -np.inf
    best_path = None

    ckpts = sorted(
        [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith(".zip") and f.startswith("ppo_eurusd")],
        key=lambda x: os.path.getmtime(os.path.join(CHECKPOINT_DIR, x))
    )

    for ck in ckpts:
        ck_path = os.path.join(CHECKPOINT_DIR, ck)
        try:
            m = PPO.load(ck_path, env=test_eval_env)
            _, final_eq = evaluate_model(m, test_eval_env)
            print(f"[OOS Eval] {ck} -> final equity: {final_eq:.2f}")
            if final_eq > best_equity:
                best_equity = final_eq
                best_path = ck_path
        except Exception as e:
            print(f"[Skip] Could not evaluate checkpoint {ck}: {e}")

    # Decide best model
    if best_path is None or final_equity_test_last >= best_equity:
        print("Using last model as best (by OOS final equity).")
        best_model = model
    else:
        print(f"Using best checkpoint: {best_path} (OOS final equity: {best_equity:.2f})")
        best_model = PPO.load(best_path, env=train_vec_env)

    best_model_path = os.path.join(MODEL_DIR, "best_model")
    best_model.save(best_model_path)
    print(f"Best model saved: {best_model_path}")

    # ---- Plot BOTH: in-sample vs out-of-sample ----
    equity_curve_train, final_equity_train = evaluate_model(best_model, train_eval_env)
    equity_curve_test, final_equity_test = evaluate_model(best_model, test_eval_env)

    print(f"[IS Eval]  Final equity (train): {final_equity_train:.2f}")
    print(f"[OOS Eval] Final equity (test) : {final_equity_test:.2f}")

    plt.figure(figsize=(12, 6))
    plt.plot(equity_curve_train, label="Train (in-sample) equity")
    plt.plot(equity_curve_test, label="Test (out-of-sample) equity")
    plt.title("Equity Curves: In-sample vs Out-of-sample (Best Model)")
    plt.xlabel("Steps")
    plt.ylabel("Equity ($)")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
