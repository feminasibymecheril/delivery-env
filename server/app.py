from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from env import DeliveryEnv, Action
from tasks import TASKS

app = FastAPI()

# ----------------------------
# Build TASKS_BY_ID from tasks.py
# ----------------------------
TASKS_BY_ID = {t["id"]: t for t in TASKS}

# ----------------------------
# Session storage
# ----------------------------
sessions: dict = {}
session_counter = 0

# ----------------------------
# ROOT
# ----------------------------
@app.get("/")
def root():
    return {"status": "DeliveryEnv running"}

# ----------------------------
# HEALTH
# ----------------------------
@app.get("/health")
def health():
    return {"status": "ok"}

# ----------------------------
# TASKS (FIXED)
# ----------------------------
@app.get("/tasks")
def list_tasks():
    return [
        {
            "task_id": t["id"],  # IMPORTANT FIX
            "description": t.get("description", ""),
            "num_packages": t.get("num_packages", 1),
            "num_obstacles": t.get("num_obstacles", 0),
            "max_steps": t.get("max_steps", 50),
            "has_grader": True,
        }
        for t in TASKS
    ]

# ----------------------------
# RESET
# ----------------------------
@app.post("/reset")
def reset(task_id: str = "easy"):
    global session_counter

    task = TASKS_BY_ID.get(task_id)
    if not task:
        raise HTTPException(status_code=400, detail=f"Unknown task_id '{task_id}'")

    env = DeliveryEnv()
    obs = env.reset()

    session_counter += 1
    session_id = str(session_counter)

    sessions[session_id] = {"env": env, "task": task}

    return {
        "session_id": session_id,
        "task_id": task_id,
        "observation": obs.dict()
    }

# ----------------------------
# STEP
# ----------------------------
@app.post("/step")
def step(session_id: str, action: Action):
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    env = session["env"]
    task = session["task"]

    result = env.step(action)

    grader_state = {
        "delivered_count": env.delivered_count,
        "total_packages": env.num_packages,
        "steps_used": env.steps,
        "max_steps": env.max_steps,
    }

    if result.done:
        result.reward = task["grader"](grader_state)
        del sessions[session_id]

    return {
        "observation": result.observation.dict(),
        "reward": result.reward,
        "done": result.done,
        "info": result.info,
    }

# ----------------------------
# STATE
# ----------------------------
@app.get("/state")
def state(session_id: str):
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session["env"].state().dict()

# ----------------------------
# GRADER
# ----------------------------
@app.post("/grader")
def run_grader(task_id: str, grader_state: dict):
    task = TASKS_BY_ID.get(task_id)
    if not task:
        raise HTTPException(status_code=400, detail="Invalid task_id")

    score = task["grader"](grader_state)
    return {"task_id": task_id, "score": score}

# ----------------------------
# ENTRY POINT
# ----------------------------
def main():
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()