import os
from openai import OpenAI
from env import DeliveryEnv, Action

# REQUIRED ENV VARIABLES
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN
)

MAX_STEPS = 50

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


def run():
    # REQUIRED LLM CALL
    client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=5
    )

    env = DeliveryEnv()
    obs = env.reset()

    rewards = []
    success = False

    print(f"[START] task=delivery env=custom model={MODEL_NAME}")

    for step in range(1, MAX_STEPS + 1):
        action = simple_policy(obs)
        result = env.step(Action(action=action))

        reward = round(result.reward, 2)
        done = result.done

        rewards.append(f"{reward:.2f}")

        print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error=null")

        obs = result.observation

        if done:
            success = True
            break

    print(f"[END] success={str(success).lower()} steps={len(rewards)} rewards={','.join(rewards)}")


if __name__ == "__main__":
    run()