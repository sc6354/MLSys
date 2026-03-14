"""
Diagnose why subgraphs couldn't be merged in the output schedule.
Usage: python diagnose.py benchmarks/mlsys-2026-17.json output_17.json
"""
import sys, json

def diagnose(input_path, output_path):
    with open(input_path) as f:
        problem = json.load(f)
    with open(output_path) as f:
        schedule = json.load(f)

    widths   = problem["widths"]
    heights  = problem["heights"]
    inputs   = problem["inputs"]
    outputs  = problem["outputs"]
    op_types = problem["op_types"]
    capacity = problem["fast_memory_capacity"]
    native_w, native_h = problem["native_granularity"]

    # Build tensor -> producer map
    tensor_producer = {}
    for op in range(len(op_types)):
        for t in outputs[op]:
            tensor_producer[t] = op

    all_tensors  = set(range(len(widths)))
    graph_inputs = all_tensors - set(tensor_producer.keys())

    # Identify pinned tensors (graph inputs used by 3+ ops)
    use_count = {}
    for op in range(len(op_types)):
        for t in inputs[op]:
            if t in graph_inputs:
                use_count[t] = use_count.get(t, 0) + 1
    pinned = {t for t, c in use_count.items() if c > 2}
    pinned_size = sum(widths[t] * heights[t] for t in pinned)

    subgraphs = schedule["subgraphs"]
    latencies = schedule.get("subgraph_latencies") or [0.0] * len(subgraphs)

    print(f"\nBenchmark: {input_path}")
    print(f"Total subgraphs: {len(subgraphs)}, Total latency: {sum(latencies):.2f}")
    print(f"Fast memory capacity: {capacity}, Pinned tensors: {pinned}, Pinned size: {pinned_size}")
    print("\n--- Per-subgraph analysis ---\n")

    for i, sg in enumerate(subgraphs):
        produced = set()
        consumed = set()
        for op in sg:
            for t in outputs[op]: produced.add(t)
            for t in inputs[op]:  consumed.add(t)
        ephemeral    = produced & consumed
        boundary_in  = consumed - produced - pinned
        boundary_out = produced - ephemeral
        working_set  = (len(boundary_in) + len(boundary_out)) * native_w * native_h + pinned_size

        print(f"Subgraph {i:2d} | ops={sg} | latency={latencies[i]:.1f}")
        print(f"           boundary_in={sorted(boundary_in)} boundary_out={sorted(boundary_out)}")
        print(f"           working_set={working_set} / {capacity} ({'OK' if working_set <= capacity else 'OOM'})")

        # Try merging with next subgraph and explain why it fails
        if i + 1 < len(subgraphs):
            next_sg = subgraphs[i + 1]
            merged  = sg + next_sg

            # Check memory
            mp, mc = set(), set()
            for op in merged:
                for t in outputs[op]: mp.add(t)
                for t in inputs[op]:  mc.add(t)
            me  = mp & mc
            mbi = mc - mp - pinned
            mbo = mp - me
            mws = (len(mbi) + len(mbo)) * native_w * native_h + pinned_size

            # Check topo: any op in next_sg needs a tensor produced outside merged?
            produced_so_far = set()
            for prev_sg in subgraphs[:i]:
                for op in prev_sg:
                    for t in outputs[op]: produced_so_far.add(t)
            for op in sg:
                for t in outputs[op]: produced_so_far.add(t)

            topo_blockers = []
            for op in next_sg:
                for t in inputs[op]:
                    if (t in tensor_producer
                            and tensor_producer[t] not in merged
                            and t not in produced_so_far
                            and t not in pinned):
                        topo_blockers.append((op, t, tensor_producer[t]))

            if mws > capacity:
                print(f"           ✗ CANNOT merge with Subgraph {i+1}: working_set={mws} > capacity={capacity}")
            elif topo_blockers:
                print(f"           ✗ CANNOT merge with Subgraph {i+1}: topo blockers {topo_blockers}")
            else:
                print(f"           ✓ COULD merge with Subgraph {i+1} (merged working_set={mws})")
        print()

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python diagnose.py <input.json> <output.json>")
    else:
        diagnose(sys.argv[1], sys.argv[2])
