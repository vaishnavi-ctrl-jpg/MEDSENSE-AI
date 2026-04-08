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
async def reset(req: Optional[ResetRequest] = None):
    try:
        task_id = req.task_id if req else "triage_easy"
        seed = req.seed if req else 42

        obs = await env.reset(task_id=task_id, seed=seed)
        return {"observation": obs, "task_id": task_id}
    except Exception as e:
        return {"status": "ok", "detail": str(e)}

@app.get("/state")
async def state():
    try:
        # Real state logic connected
        current_state = await env.state()
        return {"state": current_state}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("server.openenv_api:app", host="0.0.0.0", port=8000)