"""
MedSense AI — Start Script
Run: python start.py
Opens API + browser automatically.
"""
import sys, os, threading, webbrowser, time, subprocess

def start_api():
    os.system(f"{sys.executable} backend/api.py")

if __name__ == "__main__":
    print("\n  Starting MedSense AI...")
    t = threading.Thread(target=start_api, daemon=True)
    t.start()
    time.sleep(2)
    ui = os.path.abspath("frontend/index.html")
    webbrowser.open(f"file:///{ui}")
    print("  API  → http://localhost:5000/api/health")
    print("  UI   → frontend/index.html")
    print("\n  Press Ctrl+C to stop.\n")
    try: t.join()
    except KeyboardInterrupt: print("\n  Stopped.")
