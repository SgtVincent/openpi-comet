#!/usr/bin/env python3
"""Analyze eval log to extract startup timing breakdown."""
import sys
import re
from pathlib import Path

def analyze_log(log_path: str):
    text = Path(log_path).read_text(errors="ignore")
    lines = text.splitlines()

    events = []
    for i, line in enumerate(lines, 1):
        # Match timestamps like [00:00:24.208] (OG format) or [49.840s] (Isaac Sim format)
        m = re.search(r'\[(\d{2}):(\d{2}):(\d{2}\.\d+)\]', line)
        if m:
            h, mi, s = int(m.group(1)), int(m.group(2)), float(m.group(3))
            ts = h * 3600 + mi * 60 + s
            events.append((ts, i, line.strip()))
            continue
        m2 = re.search(r'\[(\d+\.\d+)s\]', line)
        if m2:
            ts = float(m2.group(1))
            events.append((ts, i, line.strip()))

    if not events:
        print("No timestamped events found.")
        return

    # Key milestones
    milestones = {
        "Starting OmniGibson": None,
        "Failed to solve some dependencies": None,
        "Simulation App Starting": None,
        "Simulation App Startup Complete": None,
        "Welcome to OmniGibson": None,
        "Connected to server": None,
        "Starting evaluation": None,
        "Starting task instance": None,
    }

    for ts, ln, line in events:
        for key in milestones:
            if milestones[key] is None and key.lower() in line.lower():
                milestones[key] = (ts, ln, line)
                break

    print(f"=== Startup Timing Analysis: {log_path} ===\n")
    for key, val in milestones.items():
        if val:
            ts, ln, _ = val
            print(f"  {key:40s} @ {ts:8.2f}s  (line {ln})")
        else:
            print(f"  {key:40s}  NOT FOUND")

    # Compute deltas
    def delta(a, b):
        if milestones[a] and milestones[b]:
            return milestones[b][0] - milestones[a][0]
        return None

    print("\n--- Deltas ---")
    d = delta("Starting OmniGibson", "Simulation App Starting")
    if d: print(f"  OmniGibson init -> Sim App Start      : {d:.1f}s")
    d = delta("Simulation App Starting", "Simulation App Startup Complete")
    if d: print(f"  Sim App Start -> Startup Complete     : {d:.1f}s  <-- ISAAC SIM COLD START")
    d = delta("Simulation App Startup Complete", "Welcome to OmniGibson")
    if d: print(f"  Startup Complete -> Welcome OG        : {d:.1f}s")
    d = delta("Welcome to OmniGibson", "Connected to server")
    if d: print(f"  Welcome OG -> Connected to server     : {d:.1f}s  <-- SCENE LOADING")
    d = delta("Connected to server", "Starting evaluation")
    if d: print(f"  Connected -> Starting evaluation      : {d:.1f}s")
    d = delta("Starting evaluation", "Starting task instance")
    if d: print(f"  Starting eval -> First task instance  : {d:.1f}s")

    total = delta("Starting OmniGibson", "Starting task instance")
    if total:
        print(f"\n  TOTAL (OmniGibson -> First task)      : {total:.1f}s")

    # Count hash mismatch warnings
    hash_warnings = text.count("was expected to have USD file hash")
    print(f"\n  USD hash mismatch warnings: {hash_warnings}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_startup_time.py <eval_log_path>")
        sys.exit(1)
    analyze_log(sys.argv[1])
