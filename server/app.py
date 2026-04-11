from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from env import DeliveryEnv, Action, TASKS, TASKS_BY_ID

app = FastAPI()

# ----------------------------
# Session storage (per reset)
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
# HEALTH (required by validator)
# ----------------------------
@app.get("/health")
def health():
    return {"status": "ok"}

# ----------------------------
# TASKS (required by validator)
# ----------------------------
@app.get("/tasks")
def list_tasks():
    return [
        {
            "task_id": t["task_id"],
            "description": t["description"],
            "num_packages": t["num_packages"],
            "num_obstacles": t["num_obstacles"],
            "max_steps": t["max_steps"],
            "has_grader": True,   # ← validator checks this field
        }
        for t in TASKS
    ]

# ----------------------------
# RESET (now supports task_id)
# ----------------------------
@app.post("/reset")
def reset(task_id: str = "easy"):
    global session_counter

    task = TASKS_BY_ID.get(task_id)
    if not task:
        raise HTTPException(status_code=400, detail=f"Unknown task_id '{task_id}'. Choose from: easy, medium, hard")

    env = DeliveryEnv(
        num_packages=task["num_packages"],
        num_obstacles=task["num_obstacles"],
        max_steps=task["max_steps"],
    )
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
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found. Call /reset first.")

    env = session["env"]
    task = session["task"]
    result = env.step(action)

    # Build grader state for final scoring
    grader_state = {
        "delivered_count": env.delivered_count,
        "total_packages": env.num_packages,
        "steps_used": env.steps,
        "max_steps": env.max_steps,
    }

    # On episode end, replace reward with grader score
    if result.done:
        result.reward = task["grader"](grader_state)
        # Clean up session
        del sessions[session_id]

    return {
        "observation": result.observation.dict(),
        "reward": result.reward,
        "done": result.done,
        "info": result.info,
    }

# ----------------------------
# STATE (required by OpenEnv)
# ----------------------------
@app.get("/state")
def state(session_id: str):
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    return session["env"].state().dict()

# ----------------------------
# GRADER (required by validator)
# ----------------------------
@app.post("/grader")
def run_grader(task_id: str, grader_state: dict):
    task = TASKS_BY_ID.get(task_id)
    if not task:
        raise HTTPException(status_code=400, detail=f"Unknown task_id '{task_id}'")
    score = task["grader"](grader_state)
    return {"task_id": task_id, "score": score}

# ----------------------------
# ENTRY POINT
# ----------------------------
def main():
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()
