import os
import re
import numpy as np
import matplotlib.pyplot as plt

LOG_DIR = "logs"
MOVING_AVG_WINDOW = 50  # 이동 평균 길이


def load_episode_rewards(log_dir):
    episodes = []
    rewards = []

    for filename in os.listdir(log_dir):
        if not filename.startswith("episode_") or not filename.endswith(".txt"):
            continue

        match = re.search(r"episode_(\d+)\.txt", filename)
        if not match:
            continue

        episode_num = int(match.group(1))

        with open(os.path.join(log_dir, filename), "r") as f:
            reward = None
            for line in f:
                if "Reward:" in line:
                    reward = float(line.split("Reward:")[1].strip())
                    break

        if reward is not None:
            episodes.append(episode_num)
            rewards.append(reward)

    sorted_pairs = sorted(zip(episodes, rewards), key=lambda x: x[0])
    episodes, rewards = zip(*sorted_pairs)
    return np.array(episodes), np.array(rewards)

def plot_episode_rewards():
    episodes, rewards = load_episode_rewards(LOG_DIR)

    plt.figure(figsize=(12, 6))

    # ① 모든 에피소드 점(산점도)
    plt.scatter(episodes, rewards, s=12, c="blue", alpha=0.5, label="Episode Reward (scatter)")

    # ② 모든 에피소드 연결 선(line)
    plt.plot(episodes, rewards, color="blue", alpha=0.3, linewidth=1.0, label="Episode Reward (line)")

    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Episode Reward Trend (Scatter + Line + Moving Average)")
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    plot_episode_rewards()
