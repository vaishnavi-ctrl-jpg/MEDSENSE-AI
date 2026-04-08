from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any, Dict, Optional
import uvicorn

# TODO: Uncomment this exactly when ViVi finishes her wrapper!
# from medsense.openenv_wrapper import MedSenseOpenEnv

app = FastAPI(title="MedSense AI - OpenEnv API")

# Initialize ViVi's environment wrapper (Uncomment when ready)
# env = MedSenseOpenEnv()

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
        # TODO: Uncomment ViVi's async call when ready
        # obs = await env.reset(task_id=req.task_id, seed=req.seed)
        
        # Temporary mock response so you can test the API right now
        obs = {"status": "mock_reset_ready", "task": req.task_id}
        
        return {"observation": obs, "task_id": req.task_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/step")
async def step(req: StepRequest):
    try:
        # TODO: Uncomment ViVi's async call when ready
        # step_result = await env.step(req.action)
        
        # Temporary mock response 
        step_result = {
            "observation": {"status": "mock_step_taken"},
            "reward": 1.0, 
            "done": False, 
            "info": {"parsed_action": req.action}
        }
        return step_result
    except Exception as e:
        # If ViVi's parsing fails, it gets caught here
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/state")
async def state():
    try:
        # TODO: Uncomment ViVi's async call when ready
        # current_state = await env.state()
        
        return {"state": {"mock_vitals": "stable", "queue": 0}}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("server.openenv_api:app", host="0.0.0.0", port=8000, reload=True)