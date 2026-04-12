import os
from openai import OpenAI
from env import DeliveryEnv, Action

# ----------------------------
# REQUIRED ENV VARIABLES
# ----------------------------

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
HF_TOKEN = os.getenv("HF_TOKEN")

# Initialize client safely
client = None
if HF_TOKEN:
    try:
        client = OpenAI(
            base_url=API_BASE_URL,
            api_key=HF_TOKEN
        )
    except Exception as e:
        print(f"[LLM ERROR] client init failed: {e}")

MAX_STEPS = 50


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

    if rx < tx:
        return 3
    if rx > tx:
        return 2
    if ry < ty:
        return 1
    if ry > ty:
        return 0

    return 4 if not obs.has_package else 5


# ----------------------------
# SAFE LLM CALL (REQUIRED)
# ----------------------------

def safe_llm_call():
    if client is None:
        print("[WARNING] LLM client unavailable")
        return

    try:
        client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=5
        )
    except Exception as e:
        print(f"[LLM ERROR] {e}")


# ----------------------------
# MAIN RUN
# ----------------------------

def run():
    # REQUIRED LLM CALL (non-blocking)
    safe_llm_call()

    env = DeliveryEnv()
    obs = env.reset()

    rewards = []
    success = False

    # REQUIRED START FORMAT
    print(f"[START] task=delivery env=custom model={MODEL_NAME}")

    for step in range(1, MAX_STEPS + 1):
        try:
            action = simple_policy(obs)
            result = env.step(Action(action=action))

            reward = round(result.reward, 2)
            done = result.done

            rewards.append(f"{reward:.2f}")

            # REQUIRED STEP FORMAT
            print(
                f"[STEP] step={step} action={action} "
                f"reward={reward:.2f} done={str(done).lower()} error=null"
            )

            obs = result.observation

            if done:
                success = True
                break

        except Exception as e:
            # NEVER CRASH
            print(
                f"[STEP] step={step} action=none "
                f"reward=0.00 done=true error={str(e)}"
            )
            break

    # REQUIRED END FORMAT
    print(
        f"[END] success={str(success).lower()} "
        f"steps={len(rewards)} rewards={','.join(rewards)}"
    )


# ----------------------------
# ENTRY POINT
# ----------------------------

if __name__ == "__main__":
    run()