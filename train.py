import os

from gymnasium.wrappers import RecordEpisodeStatistics
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from app import TeamFightEnv
from logging_callback import EpisodeLoggingCallback


def make_env():
    def _init():
        env = TeamFightEnv()
        env = RecordEpisodeStatistics(env)
        return env

    return _init

def main():
    env = SubprocVecEnv([make_env() for _ in range(8)])

    MODEL_PATH = "teamfight_ppo.zip"

    callback = EpisodeLoggingCallback(
        log_dir="logs",
        record_every=10,  # N회마다 기록
        verbose=1
    )

    if os.path.exists(MODEL_PATH):
        print("📌 기존 모델을 불러와서 이어서 학습합니다.")
        model = PPO.load(MODEL_PATH, env=env)
    else:
        print("📌 새 모델을 생성합니다.")
        model = PPO(
            "MlpPolicy",
            env,
            n_steps=2048,
            batch_size=64,
            learning_rate=3e-4,
            gamma=0.95,
            verbose=1,
        )

    # 이어서 학습
    model.learn(total_timesteps=1_000_000, callback=callback)

    # 저장
    model.save("teamfight_ppo")

if __name__ == "__main__":
    main()
