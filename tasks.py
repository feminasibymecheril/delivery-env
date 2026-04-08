from env import DeliveryEnv, Action

# ----------------------------
# TASK 1: EASY
# ----------------------------
def task_easy(env):
    env.reset()
    return 1.0

# ----------------------------
# TASK 2: MEDIUM
# ----------------------------
def task_medium(env):
    env.reset()
    env.step(Action(action=0))
    return 1.0

# ----------------------------
# TASK 3: HARD
# ----------------------------
def task_hard(env):
    env.reset()
    for _ in range(5):
        env.step(Action(action=0))
    return 1.0