"""
MedSense AI — Flask REST API
FIX: Removed flask_cors — uses native Flask CORS headers instead.
Run: python backend/api.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from flask import Flask, request, jsonify
import numpy as np, json
from medsense.triage_env import MedSenseEnv
from medsense.grader import TriageGrader, random_agent, rule_based_agent
from medsense.models import ACTION_NAMES, COMPLAINTS

app = Flask(__name__)

@app.after_request
def cors(r):
    r.headers["Access-Control-Allow-Origin"]  = "*"
    r.headers["Access-Control-Allow-Headers"] = "Content-Type"
    r.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return r

@app.route("/api/<path:p>", methods=["OPTIONS"])
def opts(p): return jsonify({}), 200

_env=None; _ep_r=0.0; _eps=0; _wins=0; _decisions=0; _cmiss=0
_history=[]; _acts={"treat_now":0,"delay":0,"refer":0}; _log=[]

def _pat():
    if not _env or not _env._current_patient: return {}
    p=_env._current_patient; o=p.observation
    hist=[x for x,f in[("HTN/Cardiac",o.has_cardiac_history),("Diabetes",o.has_diabetes),("Respiratory",o.has_respiratory_disease)] if f]
    return {"name":p.name,"age":p.age,"gender":p.gender,"severity":p.true_severity,
            "bp_systolic":round(o.bp_systolic,1),"bp_diastolic":round(o.bp_diastolic,1),
            "heart_rate":round(o.heart_rate,1),"spo2":round(o.spo2,1),
            "temperature":round(o.temperature,1),"resp_rate":round(o.resp_rate,1),
            "complaint":o.chief_complaint,"complaint_text":COMPLAINTS.get(o.chief_complaint,"?"),
            "pain_score":o.pain_score,"has_cardiac":o.has_cardiac_history,
            "has_diabetes":o.has_diabetes,"queue_length":o.queue_length,
            "history":", ".join(hist) if hist else "No significant history",
            "reasoning":p.reasoning,"correct_action":p.correct_action}

def _meta():
    wr=round(_wins/max(1,_eps)*100,1); cm=round(_cmiss/max(1,_decisions)*100,1)
    ar=round(sum(_history[-20:])/max(1,len(_history[-20:])),1) if _history else 0
    return {"total_episodes":_eps,"total_wins":_wins,"win_rate":wr,"avg_reward":ar,
            "episode_reward":round(_ep_r,2),"crit_miss_rate":cm,
            "action_counts":_acts,"reward_history":_history[-50:]}

@app.route("/api/health")
def health(): return jsonify({"status":"ok","message":"MedSense API running"})

@app.route("/api/tasks")
def tasks():
    return jsonify([
        {"id":"triage_easy","label":"Easy — Single Patient","threshold":"85%","patients":1},
        {"id":"triage_medium","label":"Medium — Ambiguous Case","threshold":"72%","patients":1},
        {"id":"triage_hard","label":"Hard — Multi-Patient Queue","threshold":"60%","patients":5},
    ])

@app.route("/api/reset", methods=["POST"])
def reset():
    global _env,_ep_r,_acts,_log,_decisions,_cmiss
    d=request.get_json(silent=True) or {}
    task=d.get("task_id","triage_easy"); seed=d.get("seed",42)
    _env=MedSenseEnv(task_id=task); obs,info=_env.reset(seed=seed)
    _ep_r=0.0; _acts={"treat_now":0,"delay":0,"refer":0}; _log=[]
    return jsonify({"observation":obs.tolist(),"task_id":task,
                    "patient":_pat(),"queue_size":info.get("patients_in_queue",1),"meta":_meta()})

@app.route("/api/step", methods=["POST"])
def step():
    global _env,_ep_r,_eps,_wins,_decisions,_cmiss
    if not _env: return jsonify({"error":"Call /api/reset first"}),400
    d=request.get_json(silent=True) or {}; ai=d.get("action",1)
    if isinstance(ai,str): action={"treat_now":0,"delay":1,"refer":2}.get(ai.lower(),1)
    else: action=int(ai)
    if action not in [0,1,2]: return jsonify({"error":f"Invalid action {action}"}),400
    obs,r,term,trunc,info=_env.step(action)
    _ep_r+=r; _acts[ACTION_NAMES[action]]+=1; _decisions+=1
    if info.get("critical_miss"): _cmiss+=1
    done=term or trunc
    if done:
        _eps+=1
        if info.get("won"): _wins+=1
        _history.append(round(_ep_r,2))
    return jsonify({"observation":obs.tolist(),"reward":round(r,2),"done":done,
                    "won":info.get("won",False),"action_name":ACTION_NAMES[action],
                    "correct":info.get("correct",False),"critical_miss":info.get("critical_miss",False),
                    "explanation":info.get("explanation",""),"patient":_pat() if not done else {},
                    "meta":_meta()})

@app.route("/api/grade", methods=["POST"])
def grade():
    d=request.get_json(silent=True) or {}
    task=d.get("task_id","triage_easy"); n=min(int(d.get("n_episodes",50)),200)
    policy=rule_based_agent if d.get("agent","rule_based")=="rule_based" else random_agent
    g=TriageGrader(n_episodes=n,seed=42); rep=g.evaluate(policy,task_id=task)
    return jsonify({"task_id":task,"task_passed":rep.task_passed,
                    "win_rate":round(rep.win_rate*100,1),"avg_accuracy":round(rep.avg_accuracy*100,1),
                    "avg_crit_miss":round(rep.avg_critical_miss_rate*100,1),
                    "avg_reward":round(rep.avg_reward,2),"n_episodes":n})

@app.route("/api/leaderboard")
def leaderboard():
    g=TriageGrader(n_episodes=30,seed=42); out={}
    for t in ["triage_easy","triage_medium","triage_hard"]:
        for nm,pol in [("random",random_agent),("rule_based",rule_based_agent)]:
            r=g.evaluate(pol,t)
            out[f"{nm}_{t}"]={"agent":nm,"task_id":t,"task_passed":r.task_passed,
                "win_rate":round(r.win_rate*100,1),"avg_accuracy":round(r.avg_accuracy*100,1),
                "avg_crit_miss":round(r.avg_critical_miss_rate*100,1),"avg_reward":round(r.avg_reward,2)}
    return jsonify(out)


@app.route("/api/algorithms")
def algorithms():
    agents_dir = os.path.join(os.path.dirname(__file__), "..", "agents")
    available = ["random", "rule_based"]
    for task in ["triage_easy", "triage_medium", "triage_hard"]:
        if os.path.exists(os.path.join(agents_dir, f"medsense_dqn_{task}.pth")):
            available.append(f"dqn_{task}")
        if os.path.exists(os.path.join(agents_dir, f"medsense_ppo_{task}.pth")):
            available.append(f"ppo_{task}")
    return jsonify({"available": available})

@app.route("/api/grade/all", methods=["POST"])
def grade_all():
    d = request.get_json(silent=True) or {}
    n = min(int(d.get("n_episodes", 30)), 100)
    policy = rule_based_agent if d.get("agent","rule_based") == "rule_based" else random_agent
    g = TriageGrader(n_episodes=n, seed=42)
    results = {}; all_passed = True
    for task in ["triage_easy", "triage_medium", "triage_hard"]:
        rep = g.evaluate(policy, task)
        results[task] = {"task_passed": rep.task_passed,
                         "win_rate": round(rep.win_rate*100,1),
                         "avg_accuracy": round(rep.avg_accuracy*100,1),
                         "avg_crit_miss": round(rep.avg_critical_miss_rate*100,1),
                         "avg_reward": round(rep.avg_reward,2)}
        if not rep.task_passed: all_passed = False
    return jsonify({"all_tasks_passed": all_passed, "results": results})

@app.route("/api/results")
def results_endpoint():
    path = os.path.join(os.path.dirname(__file__), "..", "results", "comparison_results.json")
    if os.path.exists(path):
        with open(path) as f: return jsonify(json.load(f))
    return jsonify({"message": "Run: python agents/compare_agents.py", "available": False})

@app.route("/api/clinical")
def clinical():
    try:
        from medsense.clinical_references import CLINICAL_REFERENCES, TRIAGE_SYSTEM_BASIS
        return jsonify({"triage_system": TRIAGE_SYSTEM_BASIS["system"],
                        "sources": TRIAGE_SYSTEM_BASIS["sources"],
                        "n_thresholds": len(CLINICAL_REFERENCES)})
    except: return jsonify({"error": "Not available"}), 500

if __name__=="__main__":
    print("\n"+"="*50+"\n  MedSense API → http://localhost:5000\n"+"="*50+"\n")
    app.run(debug=False,port=5000,threaded=True)