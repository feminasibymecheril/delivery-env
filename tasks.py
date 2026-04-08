from env import DeliveryEnv, Action

# ----------------------------
# TASK 1: EASY
# ----------------------------
def task_easy():
    env = DeliveryEnv()
    state = env.reset()

    # Just check if reset works
    return 1.0

# ----------------------------
# TASK 2: MEDIUM
# ----------------------------
def task_medium():
    env = DeliveryEnv()
    state = env.reset()

    # Take one step
    result = env.step(Action(action=0))

    return 1.0

# ----------------------------
# TASK 3: HARD
# ----------------------------
def task_hard():
    env = DeliveryEnv()
    state = env.reset()

    # Run few steps
    for _ in range(5):
        result = env.step(Action(action=0))

    return 1.0