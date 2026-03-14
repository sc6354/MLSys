# Overview

This repo contains Susan's submission to Track B for Google MLSys Competition Track Problems on https://github.com/yarongmu-google/MLSys/tree/main.

# Project: The Digital Traffic Controller (ML Systems Optimization)

### **The Challenge: Managing a Complex Digital Warehouse**
Imagine a massive, high-tech warehouse (a Machine Learning model) with many different workstations (nodes). Every station must complete a specific task and then pass its "work box" (data) to the next station.

Track B Problem - The warehouse has limited space on its conveyor belts (**memory**) and a strict deadline to get everything out the door (**speed**). If the boxes are too big or move too slowly, the whole system jams.

---
### **The Tool: A "Tensors" Primer**
To understand the solution, we first have to look at what is inside those "work boxes":
* **The Tensor:** A specialized container for information — think of it as a multi-dimensional crate. While a standard spreadsheet is flat, a Tensor can be a 3D cube of data, allowing the computer to process complex patterns like video or deep financial trends all at once.

---
### **Our Solution: A Three-Stage Approach**

**Stage 1 — The In-House Plan (Instant, no AI needed)**

Before making any AI calls, the program builds a complete working schedule using math and logic:

1. **Mapping the workflow:** Reads the entire job list and figures out the correct execution order — making sure Station B doesn't start until Station A has finished.
2. **Smart Zoning (Subgraphs):** Groups neighboring workstations together. When tasks are grouped, crates passing between them never leave the conveyor belt, saving bandwidth. High-reuse crates (tensors consumed by many stations) are pinned on a nearby shelf for the entire run.
3. **Right-sizing the workload (Granularity):** Picks the largest "bite size" that fits within the warehouse's shelf space, minimising unnecessary trips.
4. **Local Storage (Tensors to Retain):** Identifies crates the next zone will need immediately and keeps them on a nearby shelf instead of sending them to the basement.

**Stage 2 — The Optimizer (Local search)**

After the initial plan is built, the program runs a self-improvement loop for several minutes using simulated annealing — the same core technique used by the 2nd place finisher in last year's contest. It repeatedly tries three types of changes:

* **Merge** two adjacent zones into one (fewer load/evict cycles)
* **Split** an expensive zone into two smaller ones (reduces memory pressure)
* **Shift** a task from one zone to its neighbour (fine-tunes boundaries)

Crucially, it focuses its effort on the most expensive zones first — zones that account for more of the total latency get attacked more often, similar to the "eliminate expensive edge" strategy used by the 3rd place finisher.

**Stage 3 — The Expert Review (One AI call)**

Once the locally-optimized plan is ready, the program makes a single call to Google's Gemini AI. The AI receives the full problem, the optimized schedule, and detailed instructions including the exact latency formula, working set constraints, and worked examples. Its job is to look for higher-level improvements — better groupings, smarter recomputation vs. spilling decisions — and return a refined version.

If the AI is unavailable or the time limit is close, the program falls back to the Stage 2 result, which is already a strong solution.

### Benchmark 17 Execution Timeline
<p align="center">
  <img src="schedule_timeline.png" width="800">
</p>

---
### **Key Takeaways & Learnings**
* **Math first, AI second:** Most of the scheduling logic is pure arithmetic. Doing it locally is faster, free, and never hits a rate limit.
* **Focus effort on bottlenecks:** Spending search time on the most expensive subgraphs — rather than picking randomly — produces better results in the same amount of time.
* **Resilience matters:** Because the AI is optional, the solution always produces a valid answer even if the API is down or busy.
* **Efficiency at Scale:** This logic doesn't just work for small graphs — it scales to the millions of data crates moving through global systems like YouTube every second.
