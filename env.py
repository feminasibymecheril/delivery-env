from pydantic import BaseModel
from typing import List, Tuple
import random

# ----------------------------
# MODELS (OpenEnv REQUIRED)
# ----------------------------

class Observation(BaseModel):
    robot: Tuple[int, int]
    packages: List[Tuple[int, int]]
    goals: List[Tuple[int, int]]
    has_package: bool
    current_package_idx: int
    obstacles: List[Tuple[int, int]]

class Action(BaseModel):
    action: int  # 0=up, 1=down, 2=left, 3=right, 4=pick, 5=deliver

class StepResult(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: dict = {}

# ----------------------------
# ENVIRONMENT
# ----------------------------

class DeliveryEnv:
    GRID_SIZE = 5

    def __init__(self, num_packages=1, num_obstacles=0, max_steps=50):
        self.num_packages = num_packages
        self.num_obstacles = num_obstacles
        self.max_steps = max_steps
        self.delivered_count = 0
        self.reset()

    # ----------------------------
    # Helper: Unique positions
    # ----------------------------
    def get_unique_positions(self, count, exclude=None):
        if exclude is None:
            exclude = []

        positions = set()
        while len(positions) < count:
            pos = (random.randint(0, 4), random.randint(0, 4))
            if pos not in positions and pos not in exclude:
                positions.add(pos)
        return list(positions)

    # ----------------------------
    # Reset
    # ----------------------------
    def reset(self) -> Observation:
        self.steps = 0
        self.delivered_count = 0

        self.robot = (random.randint(0, 4), random.randint(0, 4))

        self.packages = self.get_unique_positions(
            self.num_packages, exclude=[self.robot]
        )

        self.goals = self.get_unique_positions(
            self.num_packages, exclude=[self.robot] + self.packages
        )

        self.obstacles = self.get_unique_positions(
            self.num_obstacles,
            exclude=[self.robot] + self.packages + self.goals
        )

        self.has_package = False
        self.current_package_idx = 0

        return self.state()

    # ----------------------------
    # State
    # ----------------------------
    def state(self) -> Observation:
        return Observation(
            robot=self.robot,
            packages=self.packages,
            goals=self.goals,
            has_package=self.has_package,
            current_package_idx=self.current_package_idx,
            obstacles=self.obstacles,
        )

    # ----------------------------
    # Step
    # ----------------------------
    def step(self, action: Action) -> StepResult:
        x, y = self.robot
        reward = -0.05
        done = False

        moves = [(0, -1), (0, 1), (-1, 0), (1, 0)]

        # Movement
        if action.action in [0, 1, 2, 3]:
            dx, dy = moves[action.action]
            nx = max(0, min(4, x + dx))
            ny = max(0, min(4, y + dy))

            if (nx, ny) in self.obstacles:
                reward -= 2
            else:
                self.robot = (nx, ny)

        # Target selection
        if not self.has_package:
            target = min(
                self.packages,
                key=lambda p: abs(self.robot[0] - p[0]) + abs(self.robot[1] - p[1])
            )
        else:
            target = self.goals[self.current_package_idx]

        # Reward shaping
        old_dist = abs(x - target[0]) + abs(y - target[1])
        new_dist = abs(self.robot[0] - target[0]) + abs(self.robot[1] - target[1])

        if new_dist < old_dist:
            reward += 1

        # Pick
        if action.action == 4 and not self.has_package:
            if self.robot in self.packages:
                self.has_package = True
                self.current_package_idx = self.packages.index(self.robot)
                reward += 10
            else:
                reward -= 1

        # Deliver
        if action.action == 5 and self.has_package:
            if self.robot == self.goals[self.current_package_idx]:
                self.has_package = False
                reward += 20
                self.current_package_idx += 1
                self.delivered_count += 1

                if self.current_package_idx >= self.num_packages:
                    done = True
            else:
                reward -= 1

        # Max steps
        self.steps += 1
        if self.steps >= self.max_steps:
            done = True

        return StepResult(
            observation=self.state(),
            reward=reward,
            done=done,
            info={
                "delivered_count": self.delivered_count,
                "steps_used": self.steps,
            }
        )