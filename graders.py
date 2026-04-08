from env import DeliveryEnv, Action

def run_episode(env):
    obs = env.reset()
    total_reward = 0

    for _ in range(50):
        rx, ry = obs.robot

        # choose target
        if not obs.has_package:
            target = obs.packages[obs.current_package_idx]
        else:
            target = obs.goals[obs.current_package_idx]

        tx, ty = target

        # movement logic
        if rx < tx:
            action = 3
        elif rx > tx:
            action = 2
        elif ry < ty:
            action = 1
        elif ry > ty:
            action = 0
        else:
            action = 4 if not obs.has_package else 5

        result = env.step(Action(action=action))
        total_reward += result.reward
        obs = result.observation

        if result.done:
            break

    return max(0.0, min(1.0, total_reward / 100))

# ----------------------------
# REQUIRED TASK GRADERS
# ----------------------------

def grade_easy():
    env = DeliveryEnv(num_packages=1, num_obstacles=1)
    return run_episode(env)

def grade_medium():
    env = DeliveryEnv(num_packages=1, num_obstacles=3)
    return run_episode(env)

def grade_hard():
    env = DeliveryEnv(num_packages=2, num_obstacles=5)
    return run_episode(env)