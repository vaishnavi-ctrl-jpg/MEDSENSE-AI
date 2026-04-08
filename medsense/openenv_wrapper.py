import re
import random
from pydantic import BaseModel
from typing import Optional, Dict, Any

# --- PYDANTIC MODELS ---
class Observation(BaseModel):
    bp_systolic: float
    heart_rate: float
    spo2: float
    temperature: float
    chief_complaint: int
    has_cardiac_history: int
    queue_length: int

class Action(BaseModel):
    patient_id: int
    decision: str

class StepResult(BaseModel):
    observation: dict
    reward: float
    done: bool
    info: dict

# --- THE ENVIRONMENT WRAPPER ---
class MedSenseOpenEnv:
    def __init__(self):
        self.current_state = {}
        self.step_count = 0
        self.max_steps = 10 # Episode ends after 10 decisions

    async def reset(self, task_id: str = "triage_easy", seed: Optional[int] = 42) -> dict:
        if seed is not None:
            random.seed(seed)
        
        self.step_count = 0
        
        # Generate new random patient vitals
        self.current_state = {
            "bp_systolic": round(random.uniform(90.0, 160.0), 1),
            "heart_rate": round(random.uniform(60.0, 120.0), 1),
            "spo2": round(random.uniform(88.0, 100.0), 1),
            "temperature": round(random.uniform(36.0, 39.5), 1),
            "chief_complaint": random.randint(0, 9),
            "has_cardiac_history": random.choice([0, 1]),
            "queue_length": 5 if task_id == "triage_hard" else 0
        }
        return self.current_state

    async def step(self, action_text: Any) -> dict:
        self.step_count += 1
        reward = 0.0
        info = {}

        # 1. PARSE LLM TEXT (e.g., "Treat patient 2" -> {"patient_id": 2, "decision": "treat"})
        text = str(action_text).lower()
        match = re.search(r"(treat_now|treat|delay|refer)\s*(?:patient)?\s*(\d+)?", text)

        if not match:
            # ViVi's task: Handle parsing errors -> penalty
            reward -= 0.1
            info["error"] = f"Parsing failed. Could not extract decision from: '{text}'"
        else:
            decision = match.group(1).replace("_now", "")
            patient_id = int(match.group(2)) if match.group(2) else 1
            
            info["parsed_action"] = {"patient_id": patient_id, "decision": decision}

            # 2. CALCULATE REWARDS (Mock logic based on YAML)
            is_critical = self.current_state["spo2"] < 92 or self.current_state["heart_rate"] > 110
            
            if is_critical and decision == "treat":
                reward += 1.0 # Correct action
            elif is_critical and decision != "treat":
                reward -= 2.0 # Missed critical
            elif not is_critical and decision == "treat":
                reward -= 0.5 # Over-triage
            else:
                reward += 1.0 # Correctly delayed/referred a stable patient

        # 3. END EPISODE CHECK
        done = self.step_count >= self.max_steps

        # Update state for next step
        self.current_state["queue_length"] = max(0, self.current_state["queue_length"] - 1)

        return {
            "observation": self.current_state,
            "reward": float(reward),
            "done": done,
            "info": info
        }

    async def state(self) -> dict:
        return {
            "current_observation": self.current_state, 
            "steps_taken": self.step_count,
            "status": "active" if self.step_count < self.max_steps else "finished"
        }