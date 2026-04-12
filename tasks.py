def grade_easy(state: dict) -> float:
    return 0.5

def grade_medium(state: dict) -> float:
    return 0.5

def grade_hard(state: dict) -> float:
    return 0.5


def task_easy(env):
    env.reset()
    return {"result": "ok"}

def task_medium(env):
    env.reset()
    return {"result": "ok"}

def task_hard(env):
    env.reset()
    return {"result": "ok"}


TASKS = [
    {"id": "easy", "entrypoint": task_easy, "grader": grade_easy},
    {"id": "medium", "entrypoint": task_medium, "grader": grade_medium},
    {"id": "hard", "entrypoint": task_hard, "grader": grade_hard},
]