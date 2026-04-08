from fastapi import FastAPI
from medsense.openenv_wrapper import MedSenseOpenEnv
import uvicorn

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


# ✅ REQUIRED BY OPENENV
def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)


# ✅ ENTRYPOINT
if __name__ == "__main__":
    main()