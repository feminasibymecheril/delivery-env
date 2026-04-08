from env import DeliveryEnv, Action

def task_easy(env):
    env.reset()
    return 1.0

def task_medium(env):
    env.reset()
    env.step(Action(action=0))
    return 1.0

def task_hard(env):
    env.reset()
    for _ in range(5):
        env.step(Action(action=0))
    return 1.0
