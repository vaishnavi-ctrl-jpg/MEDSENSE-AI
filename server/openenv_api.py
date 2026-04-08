from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any, Dict, Optional
import uvicorn

# ViVi's wrapper imported
from medsense.openenv_wrapper import MedSenseOpenEnv

app = FastAPI(title="MedSense AI - OpenEnv API")

# Initialize ViVi's environment wrapper 
env = MedSenseOpenEnv()

# --- PYDANTIC MODELS (Validation) ---
class ResetRequest(BaseModel):
    task_id: str = "triage_easy"
    seed: Optional[int] = 42

class StepRequest(BaseModel):
    # Accepts the LLM text string (e.g., "Treat patient 2") 
    # ViVi's wrapper will handle the parsing logic.
    action: Any 

# --- ENDPOINTS ---

@app.post("/reset")
async def reset(req: ResetRequest):
    try:
        # Real reset logic connected
        obs = await env.reset(task_id=req.task_id, seed=req.seed)
        return {"observation": obs, "task_id": req.task_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/step")
async def step(req: StepRequest):
    try:
        # Real step logic connected
        step_result = await env.step(req.action)
        return step_result
    except Exception as e:
        # If ViVi's parsing fails, it gets caught here
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/state")
async def state():
    try:
        # Real state logic connected
        current_state = await env.state()
        return {"state": current_state}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("server.openenv_api:app", host="0.0.0.0", port=8000, reload=True)