"""
MedSense AI — Flask REST API (FINAL SUBMISSION VERSION)
======================================================
FIXES:
- Port changed to 8000 for OpenEnv validator compatibility.
- Host set to 0.0.0.0 for Docker container accessibility.
- Added direct /reset and /step routes for fallback.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from flask import Flask, request, jsonify
import numpy as np, json
from medsense.triage_env import MedSenseEnv
from medsense.grader import TriageGrader, random_agent, rule_based_agent
from medsense.models import ACTION_NAMES, COMPLAINTS

app = Flask(__name__)

# CORS setup
@app.after_request
def cors(r):
    r.headers["Access-Control-Allow-Origin"] = "*"
    r.headers["Access-Control-Allow-Headers"] = "Content-Type"
    r.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return r

@app.route("/api/<path:p>", methods=["OPTIONS"])
def opts(p): return jsonify({}), 200

# Global State
_env=None; _ep_r=0.0; _eps=0; _wins=0; _decisions=0; _cmiss=0
_history=[]; _acts={"treat_now":0,"delay":0,"refer":0}

def _pat():
    if not _env or not _env._current_patient: return {}
    p=_env._current_patient; o=p.observation
    hist=[x for x,f in[("HTN/Cardiac",o.has_cardiac_history),("Diabetes",o.has_diabetes)] if f]
    return {"name":p.name,"severity":p.true_severity,"vitals":str(o.bp_systolic)}

def _meta():
    wr=round(_wins/max(1,_eps)*100,1)
    return {"total_episodes":_eps,"win_rate":wr,"episode_reward":round(_ep_r,2)}

@app.route("/api/health")
@app.route("/health")
def health(): return jsonify({"status":"ok","port":8000})

# CRITICAL: Added direct /reset for validator
@app.route("/reset", methods=["POST"])
@app.route("/api/reset", methods=["POST"])
def reset():
    global _env,_ep_r
    d=request.get_json(silent=True) or {}
    task=d.get("task_id","triage_easy")
    _env=MedSenseEnv(task_id=task); obs,info=_env.reset()
    _ep_r=0.0
    return jsonify({"observation":obs.tolist(),"patient":_pat(),"meta":_meta()})

# CRITICAL: Added direct /step for validator
@app.route("/step", methods=["POST"])
@app.route("/api/step", methods=["POST"])
def step():
    global _env,_ep_r,_eps,_wins
    if not _env: return jsonify({"error":"Reset first"}),400
    d=request.get_json(silent=True) or {}; ai=d.get("action",1)
    action=int(ai) if str(ai).isdigit() else {"treat_now":0,"delay":1,"refer":2}.get(str(ai).lower(),1)
    obs,r,term,trunc,info=_env.step(action)
    _ep_r+=r
    if term or trunc:
        _eps+=1
        if info.get("won"): _wins+=1
    return jsonify({"reward":round(r,2),"done":term or trunc,"meta":_meta()})

@app.route("/api/results")
def results_endpoint():
    path = os.path.join(os.path.dirname(__file__), "..", "results", "comparison_results.json")
    if os.path.exists(path):
        with open(path) as f: return jsonify(json.load(f))
    return jsonify({"available": False})

if __name__=="__main__":
    # CHANGED TO PORT 8000 AND HOST 0.0.0.0
    print("Starting MedSense API on Port 8000...")
    app.run(host='0.0.0.0', port=8000, debug=False, threaded=True)
