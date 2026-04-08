import os
from openai import OpenAI
import asyncio
from env import DeliveryEnv, Action

# ----------------------------
# REQUIRED ENV VARIABLES
# ----------------------------

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
MODEL_NAME = os.getenv("MODEL_NAME", "dummy-model")
HF_TOKEN = os.getenv("HF_TOKEN")  # optional

# OpenAI client (required by checklist)
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN
)

MAX_STEPS = 50

# ----------------------------
# REQUIRED LOG FORMAT
# ----------------------------

def log_start(task):
    print(f"[START] task={task}")

def log_step(step, action, reward, done):
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done}")

def log_end(success, steps, score):
    print(f"[END] success={success} steps={steps} score={score:.3f}")

# ----------------------------
# SIMPLE POLICY
# ----------------------------

def simple_policy(obs):
    rx, ry = obs.robot

    if not obs.has_package:
        target = obs.packages[obs.current_package_idx]
    else:
        target = obs.goals[obs.current_package_idx]

    tx, ty = target

    if rx < tx: return 3
    if rx > tx: return 2
    if ry < ty: return 1
    if ry > ty: return 0

    return 4 if not obs.has_package else 5

# ----------------------------
# MAIN RUN
# ----------------------------

async def main():
    env = DeliveryEnv()
    obs = env.reset()

    total_reward = 0
    steps_taken = 0

    log_start("delivery_task")

    for step in range(1, MAX_STEPS + 1):
        action = simple_policy(obs)

        result = env.step(Action(action=action))

        reward = result.reward
        done = result.done

        log_step(step, action, reward, done)

        total_reward += reward
        steps_taken = step
        obs = result.observation

        if done:
            break

    score = max(0.0, min(1.0, total_reward / 100))
    success = score > 0.5

    log_end(success, steps_taken, score)

if __name__ == "__main__":
    asyncio.run(main())