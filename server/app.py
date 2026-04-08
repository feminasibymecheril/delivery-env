from fastapi import FastAPI
import uvicorn
from env import DeliveryEnv, Action

app = FastAPI()
env = DeliveryEnv()

@app.get("/")
def root():
    return {"status": "DeliveryEnv running"}

@app.post("/reset")
def reset():
    obs = env.reset()
    return obs.dict()

@app.post("/step")
def step(action: Action):
    result = env.step(action)
    return {
        "observation": result.observation.dict(),
        "reward": result.reward,
        "done": result.done,
        "info": result.info
    }

# ✅ REQUIRED for OpenEnv
def main():
    uvicorn.run(app, host="0.0.0.0", port=8000)

# ✅ REQUIRED entry point
if __name__ == "__main__":
    main()