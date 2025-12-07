import math
import random
from contextlib import nullcontext
from queue import PriorityQueue
from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from data import Unit, StatType, UnitState, UnitStat

ALLY_COUNT = 5
NUM_UNITS = 10
MAP_SIZE = 256

class TeamFightEnv(gym.Env):
    metadata = {'render.modes': ['rgb_array']}
    def __init__(self):
        super().__init__()

        self.physics_dt = 0.05
        self.current_step = 0

        self.action_scheduler = PriorityQueue()

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(20,), dtype=np.float32)
        # self.action_space = spaces.Dict({
        #     "action_type": spaces.MultiDiscrete([3] * ALLY_COUNT),
        #     "move_dir": spaces.Box(low=-1.0, high=1.0, shape=(ALLY_COUNT, 2), dtype=np.float32),
        #     "attack_target": spaces.MultiDiscrete([NUM_UNITS] * ALLY_COUNT)
        # })

        self.units = self._generate_units()

        obs_dim = NUM_UNITS * (5 + NUM_UNITS)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim, ), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_step = 0
        self.action_scheduler = PriorityQueue()

        for idx, u in enumerate(self.units):
            u.state = UnitState.IDLE
            u.stats.set(StatType.HP, u.stats.get(StatType.MAX_HP))
            u.move_dir[:] = 0.0

            if u.team == 0:
                u.pos = np.random.uniform(MAP_SIZE * 0.3, MAP_SIZE * 0.4, size=2)
            else:
                u.pos = np.random.uniform(MAP_SIZE * 0.6, MAP_SIZE * 0.7, size=2)

            jitter = np.random.uniform(0.75, 1.25)
            next_step = int(u.reaction_steps * jitter)
            self.action_scheduler.put((next_step, idx))

        self.prev_hp = np.array([u.stats.get(StatType.HP) for u in self.units], dtype=np.float32)

        return self._build_obs(), {}

    def step(self, action):
        structured = self._unpack_action(action)

        self.current_step += 1

        while not self.action_scheduler.empty():
            next_step, idx = self.action_scheduler.queue[0]
            if next_step > self.current_step:
                break

            self.action_scheduler.get()
            unit = self.units[idx]

            if unit.state is UnitState.DEAD:
                continue

            if unit.team == 0:
                self.update_state(idx, structured)
            else:
                self.update_enemy_state(idx)

            jitter = np.random.uniform(0.75, 1.25)
            next_step = int(unit.reaction_steps * jitter)
            self.action_scheduler.put((self.current_step + next_step, idx))

        for idx, u in enumerate(self.units):
            if u.state is UnitState.DEAD:
                continue
            self.update_physics(idx)

        reward = self._compute_reward()
        terminated = self._check_terminated()
        truncated = False

        obs = self._build_obs()
        info = {}

        return obs, reward, terminated, truncated, info

    def update_state(self, idx, action):
        unit = self.units[idx]

        action_type = action["action_type"][idx]
        if action_type == 0:
            unit.state = UnitState.IDLE
            unit.move_dir[:] = 0.0
            return

        elif action_type == 1:
            unit.state = UnitState.MOVE
            raw = action["move_dir"][idx]
            norm = np.linalg.norm(raw)
            unit.move_dir[:] = (raw / norm) if norm > 1e-6 else 0.0
            return

        elif action_type == 2:
            target_idx = int(action["attack_target"][idx])
            if target_idx == idx:
                unit.state = UnitState.IDLE
                return

            if target_idx is None:
                unit.state = UnitState.IDLE
                return

            if target_idx < 0 or target_idx >= NUM_UNITS:
                unit.state = UnitState.IDLE
                return

            unit.state = UnitState.ATTACK
            unit.attack_target_idx = target_idx

            raw = action["move_dir"][idx]
            norm = np.linalg.norm(raw)
            unit.move_dir[:] = (raw / norm) if norm > 1e-6 else 0.0
            return

    def update_enemy_state(self, idx):
        unit = self.units[idx]

        unit.state = UnitState.MOVE

        target_idx = idx - 5
        if self.units[target_idx].state is UnitState.DEAD:
            target_idx = self._closest_ally(idx)

        if target_idx is None:
            unit.state = UnitState.IDLE
            unit.move_dir[:] = 0.0
            return

        target = self.units[target_idx]
        dist = np.linalg.norm(target.pos - unit.pos)
        attack_range = unit.stats.get(StatType.ATTACK_RANGE)

        if dist > attack_range:
            unit.state = UnitState.MOVE
            unit.move_dir[:] = (target.pos - unit.pos) / (dist + 1e-6)
        else:
            attack_prob = 0.1
            if random.random() < attack_prob:
                unit.state = UnitState.ATTACK
                unit.attack_target_idx = target_idx
            else:
                unit.state = UnitState.IDLE
            unit.move_dir[:] = 0.0

    def update_physics(self, idx):
        unit = self.units[idx]

        if unit.state is UnitState.MOVE:
            unit.pos += unit.move_dir * unit.stats.get(StatType.MOVE_SPEED) * self.physics_dt
        unit.pos = np.clip(unit.pos, 0.0, MAP_SIZE)

        if unit.state is UnitState.ATTACK:
            if unit.attack_target_idx is None:
                return

            target = self.units[unit.attack_target_idx]
            if target.state is UnitState.DEAD:
                return

            dist = np.linalg.norm(target.pos - unit.pos)
            if dist > unit.stats.get(StatType.ATTACK_RANGE):
                unit.pos += unit.move_dir * unit.stats.get(StatType.MOVE_SPEED) * self.physics_dt
                unit.pos = np.clip(unit.pos, 0.0, MAP_SIZE)
                return

            attack = unit.stats.get(StatType.ATTACK)
            defense = target.stats.get(StatType.DEFENSE)
            attack_speed = unit.stats.get(StatType.ATTACK_SPEED)
            dps = max(attack - defense, 1.0) * attack_speed * self.physics_dt

            target.stats.add(StatType.HP, -dps)

            if target.stats.get(StatType.HP) <= 0:
                target.stats.set(StatType.HP, 0)
                target.state = UnitState.DEAD
                unit.state = UnitState.IDLE

    def _generate_units(self, budget_per_unit=10.0):
        units = []

        for team in [0,1]:
            for i in range(ALLY_COUNT):
                w = np.random.dirichlet([1, 1, 1, 1, 1, 1])
                allocated = w * budget_per_unit

                stats = UnitStat({
                    StatType.MAX_HP: 50 + allocated[0] * 10,               # 50 ~ 150
                    StatType.HP: 0.0,
                    StatType.ATTACK: 10 + allocated[1] * 2,                 # 10 ~ 30
                    StatType.ATTACK_RANGE: 30.0 + allocated[2] * 2,            # 30 ~ 50
                    StatType.ATTACK_SPEED: 0.8 + allocated[3] * 0.1,        # 0.8 ~ 1.5
                    StatType.DEFENSE: 0.0 + allocated[4] * 1,               # 0 ~ 10
                    StatType.MOVE_SPEED: 10.0 + allocated[5] * 1,          # 10 ~ 20
                })

                if team == 0:
                    pos = np.random.uniform(0, MAP_SIZE * 0.3, size=2)
                else:
                    pos = np.random.uniform(MAP_SIZE * 0.7, MAP_SIZE, size=2)

                unit = Unit(team=team, stats=stats, pos=pos, move_dir=np.zeros(2, dtype=np.float32))
                units.append(unit)

        return units

    def _unpack_action(self, flat):
        flat = np.array(flat, dtype=np.float32)

        # === 1) Action Type (연속값 기반 구간 분할) ===
        at_raw = flat[0:ALLY_COUNT]  # shape = (5,)
        action_type = np.zeros(ALLY_COUNT, dtype=int)

        for i, v in enumerate(at_raw):
            if v < -0.33:
                action_type[i] = 0  # IDLE
            elif v < 0.33:
                action_type[i] = 1  # MOVE
            else:
                action_type[i] = 2  # ATTACK

        # === 2) Move Dir ===
        move_slice = flat[ALLY_COUNT:ALLY_COUNT * 3]  # 10 values
        move = move_slice.reshape(ALLY_COUNT, 2)

        # === 3) Attack Target (연속값 → 0~NUM_UNITS-1) ===
        tgt_raw = flat[ALLY_COUNT * 3:ALLY_COUNT * 4]  # shape = (5,)

        # [-1,1] → [0, NUM_UNITS)
        tgt = ((tgt_raw + 1) / 2 * NUM_UNITS).astype(int)
        tgt = np.clip(tgt, 0, NUM_UNITS - 1)

        structured = {
            "action_type": action_type,  # 0,1,2
            "move_dir": move,  # continuous
            "attack_target": tgt  # integer target index
        }

        return structured

    def _ally_units(self):
        return [u for u in self.units if u.team == 0]
    def _enemy_units(self):
        return [u for u in self.units if u.team == 1]

    def _closest_ally(self, idx) -> Optional[int]:
        allies = [i for i, u in enumerate(self.units) if u.team == 0 and u.state != UnitState.DEAD]
        if not allies:
            return None
        return min(allies, key=lambda i: np.linalg.norm(self.units[i].pos - self.units[idx].pos))

    def _build_obs(self):
        obs = []

        positions = np.array([u.pos for u in self.units])
        max_dist = math.sqrt(2 * (MAP_SIZE ** 2))

        # 거리 행렬 계산 (vectorization)
        dist_matrix = np.linalg.norm(
            positions[:, None, :] - positions[None, :, :],
            axis=2
        ) / max_dist  # 정규화

        for i, u in enumerate(self.units):
            team_flag = 1.0 if u.team == 0 else 0.0
            alive_flag = 1.0 if u.state is not UnitState.DEAD else 0.0
            hp_ratio = u.stats.get(StatType.HP) / u.stats.get(StatType.MAX_HP)

            pos_x_ratio = u.pos[0] / MAP_SIZE
            pos_y_ratio = u.pos[1] / MAP_SIZE

            # 거리 정보: i번째 행 전체
            distances = dist_matrix[i]

            unit_vec = [
                team_flag,
                alive_flag,
                hp_ratio,
                pos_x_ratio, pos_y_ratio,
                *distances  # 거리 정보 추가
            ]
            obs.extend(unit_vec)

        return np.array(obs, dtype=np.float32)

    def _compute_reward(self):
        reward = 0.0

        curr_hp = np.array([u.stats.get(StatType.HP) for u in self.units])
        max_hp = np.array([u.stats.get(StatType.MAX_HP) for u in self.units])

        hp_diff = self.prev_hp - curr_hp

        for idx, u in enumerate(self.units):
            if u.team == 0:
                reward -= hp_diff[idx] * 0.1
            else:
                reward += hp_diff[idx] * 0.1

        for idx, u in enumerate(self.units):
            if u.team == 1:
                if self.prev_hp[idx] > 0 >= curr_hp[idx]:
                    reward += 10.0

            if u.team == 0:
                if self.prev_hp[idx] > 0 >= curr_hp[idx]:
                    reward += 10.0

        if all(u.state == UnitState.DEAD for u in self._enemy_units()):
            reward += 1000.0
        if all(u.state == UnitState.DEAD for u in self._ally_units()):
            reward -= 1000.0

        self.prev_hp = curr_hp.copy()
        return reward

    def _check_terminated(self):
        all_allies_dead = all(u.state is UnitState.DEAD for u in self._ally_units())
        all_enemies_dead = all(u.state is UnitState.DEAD for u in self._enemy_units())
        return all_allies_dead or all_enemies_dead

    def render(self, mode="rgb_array"):
        import cv2

        # 메인 맵 캔버스
        canvas = np.zeros((MAP_SIZE, MAP_SIZE, 3), dtype=np.uint8)

        # ===== 유닛 표시 =====
        RADIUS = 3
        font = cv2.FONT_HERSHEY_SIMPLEX

        for idx, u in enumerate(self.units):
            if u.state == UnitState.DEAD:
                continue

            x = int(np.clip(u.pos[0], 0, MAP_SIZE - 1))
            y = int(np.clip(u.pos[1], 0, MAP_SIZE - 1))

            # 팀 색상
            color = (50, 200, 255) if u.team == 0 else (255, 80, 80)

            # ---- 사거리 표시 (연한 색) ----
            attack_range = int(u.stats.get(StatType.ATTACK_RANGE))
            range_color = (color[0] // 2, color[1] // 2, color[2] // 2)

            cv2.circle(
                canvas,
                (x, y),
                attack_range,
                range_color,
                thickness=1,
                lineType=cv2.LINE_AA
            )

            # ---- 유닛 원 ----
            cv2.circle(canvas, (x, y), RADIUS, color, -1, lineType=cv2.LINE_AA)

            # ---- 유닛 인덱스 표시 ----
            cv2.putText(canvas, str(idx), (x - 6, y - RADIUS - 2),
                        font, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

        # ===== 전광판 HUD =====
        HUD_HEIGHT = 90  # ← 기존 70보다 20 높임 (팀 체력 % 추가)
        hud = np.zeros((HUD_HEIGHT, MAP_SIZE, 3), dtype=np.uint8)

        color_ally = (50, 200, 255)
        color_enemy = (255, 80, 80)

        # ========= 팀 전체 체력 계산 =========
        total_ally_hp = sum(u.stats.get(StatType.HP) for u in self.units if u.team == 0)
        total_ally_max = sum(u.stats.get(StatType.MAX_HP) for u in self.units if u.team == 0)

        total_enemy_hp = sum(u.stats.get(StatType.HP) for u in self.units if u.team == 1)
        total_enemy_max = sum(u.stats.get(StatType.MAX_HP) for u in self.units if u.team == 1)

        ally_ratio = (total_ally_hp / max(total_ally_max, 1)) * 100
        enemy_ratio = (total_enemy_hp / max(total_enemy_max, 1)) * 100

        # ====== HUD 최상단에 %만 표시 ======
        top_y = 18
        font = cv2.FONT_HERSHEY_SIMPLEX

        # 왼쪽: 아군 체력 % (Cyan)
        cv2.putText(hud,
                    f"{ally_ratio:.1f}%",
                    (10, top_y),
                    font, 0.6, (50, 200, 255), 2, cv2.LINE_AA)

        # 오른쪽: 적군 체력 % (Red)
        cv2.putText(hud,
                    f"{enemy_ratio:.1f}%",
                    (150, top_y),  # 필요하면 x 위치 조정 가능
                    font, 0.6, (255, 80, 80), 2, cv2.LINE_AA)

        # -------------------------------
        # 기존 4줄 HP 표시
        # -------------------------------
        start_x = 5

        # 줄1: 아군 현재 HP
        cursor_x = start_x
        y1 = 40
        for u in self.units:
            if u.team == 0:
                cv2.putText(hud, str(int(u.stats.get(StatType.HP))), (cursor_x, y1),
                            font, 0.45, color_ally, 1, cv2.LINE_AA)
                cursor_x += 40

        # 줄2: 아군 최대 HP
        cursor_x = start_x
        y2 = 55
        for u in self.units:
            if u.team == 0:
                cv2.putText(hud, str(int(u.stats.get(StatType.MAX_HP))), (cursor_x, y2),
                            font, 0.45, color_ally, 1, cv2.LINE_AA)
                cursor_x += 40

        # 줄3: 적군 현재 HP
        cursor_x = start_x
        y3 = 70
        for u in self.units:
            if u.team == 1:
                cv2.putText(hud, str(int(u.stats.get(StatType.HP))), (cursor_x, y3),
                            font, 0.45, color_enemy, 1, cv2.LINE_AA)
                cursor_x += 40

        # 줄4: 적군 최대 HP
        cursor_x = start_x
        y4 = 85
        for u in self.units:
            if u.team == 1:
                cv2.putText(hud, str(int(u.stats.get(StatType.MAX_HP))), (cursor_x, y4),
                            font, 0.45, color_enemy, 1, cv2.LINE_AA)
                cursor_x += 40

        # ===== 맵 + HUD 결합 =====
        final = np.vstack([canvas, hud])

        if mode == "rgb_array":
            return final
        else:
            cv2.imshow("TeamFightEnv", final)
            cv2.waitKey(1)

