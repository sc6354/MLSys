# Overview 

This repo contains Susan's submission to Track B for Google MLSys Competition Track Problems on https://github.com/yarongmu-google/MLSys/tree/main.

# Project: The Digital Traffic Controller (ML Systems Optimization)

### **The Challenge: Managing a Complex Digital Warehouse**
Imagine a massive, high-tech warehouse (a Machine Learning model) with **80 different workstations** (nodes). Every station must complete a specific task and then pass its "work box" (data) to the next station. 

Track B Problem - The warehouse has limited space on its conveyor belts (**memory**) and a strict deadline to get everything out the door (**speed**). If the boxes are too big or move too slowly, the whole system jams.

---
### **The Tool: A "Tensors" Primer**
To understand the solution, we first have to look at what is inside those "work boxes":
* **The Tensor:** This is simply a specialized container for information. Think of it as a **multi-dimensional crate**. While a standard spreadsheet is flat, a Tensor can be a 3D cube of data, allowing the computer to process complex patterns like video or deep financial trends all at once.

---
### **Our Solution: Strategic Prompt Engineering**
Instead of building a new machine from scratch, I designed a **sophisticated instruction set (a Prompt)** that taught an AI how to act as the ultimate Warehouse Manager. 

### Benchmark 17 Execution Timeline
<p align="center">
  <img src="schedule_timeline.png" width="800">
</p>


**The Manager made three key strategic decisions:**
1.  **Smart Zoning (Subgraphs):** It grouped the 80 tasks into **15 efficient zones**. By keeping related tasks together, it minimized the time wasted moving heavy "tensors" across the warehouse.
2.  **Adjusting the Power (Granularity):** For most tasks, the AI used a standard speed setting denoted as **[64, 64, 64]**—this tells the hardware to process data in efficient, 64-unit "bites". However, for the final, complex bottleneck (**Subgraph 14**), the AI triggered a high-power setting of **[128, 128, 64]** to process larger chunks of data and finish the job faster.
3.  **Local Storage (Tensors to Retain):** The AI identified exactly which crates were too important to send back to the main basement. It kept them on a "nearby shelf" so the next workstation could grab them instantly, saving valuable seconds.

---
### **Key Takeaways & Learnings**
* **Communication is Power:** We proved that a well-crafted set of instructions (Prompt Engineering) can solve massive logistics problems just as effectively as complex custom code.
* **Identifying Bottlenecks:** The AI successfully spotted that **Subgraph 4** was a "heavy lift" area (20 tasks) and gave it the space it needed to succeed without slowing down the rest of the warehouse.
* **Efficiency at Scale:** This logic doesn't just work for 80 tasks; it can be scaled to manage the millions of data "crates" moving through global systems like **YouTube** every second.




