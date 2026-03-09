import json
import matplotlib.pyplot as plt

with open('output_17.json', 'r') as f:
    data = json.load(f)

fig, ax = plt.subplots(figsize=(12, 6), dpi=300) # High DPI for clarity

# Plotting each subgraph as a block
for i, nodes in enumerate(data['subgraphs']):
    start_node = min(nodes)
    duration = len(nodes)
    ax.barh(f"Subgraph {i}", duration, left=start_node, color='skyblue', edgecolor='black')
    
    # Label with Granularity
    gran = data['granularities'][i]
    ax.text(start_node + 0.5, i, f"Gran: {gran}", va='center', fontsize=8)

ax.set_xlabel("Node Execution Sequence")
ax.set_title("Machine Learning System Schedule - Benchmark 17")
plt.tight_layout()
plt.savefig('schedule_timeline.png', dpi=300)
plt.show()