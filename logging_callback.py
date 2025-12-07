from stable_baselines3.common.callbacks import BaseCallback
import os
import numpy as np

class EpisodeLoggingCallback(BaseCallback):

    def __init__(self, log_dir, record_every=None, verbose=0):
        super().__init__(verbose)
        self.log_dir = log_dir
        self.record_every = record_every  # ← N회마다 기록
        self.episode_rewards = []
        os.makedirs(log_dir, exist_ok=True)

    def _on_step(self) -> bool:
        # SubprocVecEnv → 여러 에피소드가 병렬 진행되므로 infos 리스트로 들어옴
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                reward = info["episode"]["r"]
                self.episode_rewards.append(reward)

                episode_idx = len(self.episode_rewards)

                # ① 특정 회차마다 출력
                if self.record_every and episode_idx % self.record_every == 0:
                    self._save_episode_log(episode_idx, reward)

        return True

    def _save_episode_log(self, episode_idx, reward):
        path = os.path.join(self.log_dir, f"episode_{episode_idx}.txt")
        with open(path, "w") as f:
            f.write(f"Episode: {episode_idx}\n")
            f.write(f"Reward: {reward}\n")

        if self.verbose > 0:
            print(f"[Callback] Episode {episode_idx} logged (reward={reward:.2f})")
