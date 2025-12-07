from stable_baselines3 import PPO
import cv2

from app import TeamFightEnv

model = PPO.load("teamfight_ppo")

env = TeamFightEnv()
obs, info = env.reset()

for _ in range(10000):
    action, _states = model.predict(obs, deterministic=False)
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        break

    frame = env.render(mode="rgb_array")
    cv2.imshow("TeamFightEnv", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    if cv2.waitKey(1) == 27:  # ESC
        break

cv2.destroyAllWindows()
