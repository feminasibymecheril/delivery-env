from fastapi import FastAPI
import uvicorn
from env import DeliveryEnv, Action

app = FastAPI()
env = DeliveryEnv()

@app.get("/")
def root():
    return {"status": "DeliveryEnv running"}

@app.get("/reset")
def reset():
    return env.reset()

@app.post("/step")
def step(action: Action):
    return env.step(action)

# ✅ REQUIRED for OpenEnv
def main():
    uvicorn.run(app, host="0.0.0.0", port=8000)

# ✅ REQUIRED entry point
if __name__ == "__main__":
    main()