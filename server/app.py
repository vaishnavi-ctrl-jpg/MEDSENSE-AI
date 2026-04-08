from fastapi import FastAPI, HTTPException
from medsense.openenv_wrapper import MedSenseOpenEnv

app = FastAPI(title="MedSense AI OpenEnv")

env = MedSenseOpenEnv()

@app.post("/reset")
async def reset():
    return await env.reset()

@app.post("/step")
async def step(action: str):
    return await env.step(action)

@app.get("/state")
async def state():
    return await env.state()