import sys, json, os, time, random, math
from google import genai
from google.genai import types

def calc_latency(problem, subgraph_ops, granularity, tensors_to_retain_prev):
    """
    Compute the total latency for one subgraph step.
    Returns (latency, working_set_size).
    """
    w, h, k = granularity
    widths   = problem["widths"]
    heights  = problem["heights"]
    inputs   = problem["inputs"]
    outputs  = problem["outputs"]
    op_types = problem["op_types"]
    base_costs = problem["base_costs"]
    bw       = problem["slow_memory_bandwidth"]
    native_w, native_h = problem["native_granularity"]

    # Collect boundary tensors (inputs loaded from slow mem, outputs evicted)
    # Ephemeral = produced AND consumed within the same subgraph
    produced = set()
    consumed = set()
    for op in subgraph_ops:
        for t in outputs[op]:
            produced.add(t)
        for t in inputs[op]:
            consumed.add(t)

    ephemeral = produced & consumed

    # Tensors loaded from slow memory (not produced in this subgraph, not retained from prev)
    load_tensors  = consumed - produced - set(tensors_to_retain_prev)
    # Tensors evicted to slow memory (produced, not ephemeral, not retained after)
    # We'll compute evict cost based on final outputs of the subgraph
    output_tensors = produced - ephemeral

    # Spatial tiles
    # Find representative output tensor size from first op
    first_op = subgraph_ops[0]
    out_t = outputs[first_op][0]
    out_w = widths[out_t]
    out_h = heights[out_t]

    tiles_w = math.ceil(out_w / w)
    tiles_h = math.ceil(out_h / h)
    n_spatial_tiles = tiles_w * tiles_h

    # k-split steps (only relevant for MatMul)
    has_matmul = any(op_types[op] == "MatMul" for op in subgraph_ops)
    if has_matmul:
        # Find the reduction dimension K from the first MatMul
        for op in subgraph_ops:
            if op_types[op] == "MatMul":
                lhs = inputs[op][0]
                k_full = widths[lhs]  # LHS width = K dimension
                break
        k_steps = math.ceil(k_full / k)
    else:
        k_steps = 1

    # Compute cost per tile (padded to native granularity)
    eff_w = max(w, native_w)
    eff_h = max(h, native_h)
    compute_per_tile = sum(base_costs[op] for op in subgraph_ops)
    # Scaling: if granularity < native, we still pay full native cost
    # If granularity > native, cost scales proportionally
    scale = (eff_w * eff_h) / (native_w * native_h)
    compute_per_tile = compute_per_tile * scale * k_steps

    # Memory cost per tile
    def tensor_tile_size(t, tw, th):
        return tw * th

    mem_in_per_tile  = sum(tensor_tile_size(t, w, h) for t in load_tensors)
    mem_out_per_tile = sum(tensor_tile_size(t, w, h) for t in output_tensors)
    mem_per_tile = (mem_in_per_tile + mem_out_per_tile) / bw

    latency_per_tile = max(compute_per_tile, mem_per_tile)
    total_latency = latency_per_tile * n_spatial_tiles

    return total_latency

# ── Greedy scheduler ───────────────────────
def greedy_schedule(problem):
    """
    Build a schedule using a greedy topological approach:
    - Pin high-reuse tensors (graph inputs used by many ops) in fast memory throughout.
    - Merge ops into subgraphs as aggressively as memory allows.
    - Use the largest granularity that fits in fast memory.
    """
    n_ops      = len(problem["op_types"])
    widths     = problem["widths"]
    heights    = problem["heights"]
    inputs     = problem["inputs"]
    outputs    = problem["outputs"]
    op_types   = problem["op_types"]
    base_costs = problem["base_costs"]
    capacity   = problem["fast_memory_capacity"]
    bw         = problem["slow_memory_bandwidth"]
    native_w, native_h = problem["native_granularity"]

    # Build dependency graph
    tensor_producer = {}  # tensor -> op that produces it
    for op in range(n_ops):
        for t in outputs[op]:
            tensor_producer[t] = op

    # Graph inputs = tensors with no producer
    all_tensors  = set(range(len(widths)))
    graph_inputs = all_tensors - set(tensor_producer.keys())

    # ── Identify high-reuse tensors to pin in fast memory ────────────────────
    # Count how many ops consume each graph input tensor
    tensor_use_count = {}
    for op in range(n_ops):
        for t in inputs[op]:
            if t in graph_inputs:
                tensor_use_count[t] = tensor_use_count.get(t, 0) + 1

    # Pin any graph input used by more than 2 ops — worth keeping resident
    pinned_tensors = {t for t, cnt in tensor_use_count.items() if cnt > 2}
    pinned_size    = sum(widths[t] * heights[t] for t in pinned_tensors)

    # Topological sort
    in_degree  = [0] * n_ops
    dependents = [[] for _ in range(n_ops)]
    for op in range(n_ops):
        for t in inputs[op]:
            if t in tensor_producer:
                parent = tensor_producer[t]
                in_degree[op] += 1
                dependents[parent].append(op)

    topo_order = []
    temp_in_degree = in_degree[:]
    queue = [op for op in range(n_ops) if temp_in_degree[op] == 0]
    while queue:
        op = queue.pop(0)
        topo_order.append(op)
        for dep in dependents[op]:
            temp_in_degree[dep] -= 1
            if temp_in_degree[dep] == 0:
                queue.append(dep)

    # ── Choose granularity: largest that fits in fast memory ─────────────────
    def choose_granularity(subgraph_ops, retained=set()):
        produced = set()
        consumed = set()
        for op in subgraph_ops:
            for t in outputs[op]: produced.add(t)
            for t in inputs[op]:  consumed.add(t)
        ephemeral    = produced & consumed
        boundary_in  = (consumed - produced) - retained   # need to load from slow mem
        boundary_out = produced - ephemeral                # need to evict to slow mem
        has_mm = any(op_types[op] == "MatMul" for op in subgraph_ops)
        k = native_w if has_mm else 1

        for divisor in [1, 2, 4, 8, 16]:
            w = max(native_w // divisor, 1)
            h = max(native_h // divisor, 1)
            ws = (len(boundary_in) + len(boundary_out)) * w * h
            if has_mm:
                ws += w * h  # accumulator
            ws += pinned_size  # pinned tensors always occupy fast memory
            if ws <= capacity:
                return [w, h, k]
        return [native_w // 16, native_h // 16, native_w]

    # ── Greedy merge: build subgraphs as large as memory allows ──────────────
    # A merge is valid if:
    #   1. Working set fits in fast memory.
    #   2. Every external input of the merged group is either a graph input
    #      (loadable) or produced by an earlier subgraph (already scheduled).
    #      Pinned tensors satisfy condition 2 automatically.

    subgraphs = [[op] for op in topo_order]

    def merged_fits(candidate_ops):
        """Check if a merged subgraph fits in fast memory."""
        produced = set()
        consumed = set()
        for op in candidate_ops:
            for t in outputs[op]: produced.add(t)
            for t in inputs[op]:  consumed.add(t)
        ephemeral    = produced & consumed
        boundary_in  = (consumed - produced) - pinned_tensors
        boundary_out = produced - ephemeral
        # Use native granularity for worst-case working set estimate
        ws = (len(boundary_in) + len(boundary_out)) * native_w * native_h + pinned_size
        return ws <= capacity

    merged = True
    while merged:
        merged = False
        new_subgraphs = []
        i = 0
        while i < len(subgraphs):
            current = subgraphs[i]
            # Try to absorb as many following subgraphs as possible
            while i + 1 < len(subgraphs):
                candidate = current + subgraphs[i + 1]

                # Check 1: merged working set fits in fast memory
                if not merged_fits(candidate):
                    break

                # Check 2: all external inputs of the next subgraph are satisfiable
                produced_so_far = set()
                for prev in new_subgraphs:
                    for op in prev:
                        for t in outputs[op]: produced_so_far.add(t)
                for op in current:
                    for t in outputs[op]: produced_so_far.add(t)

                can_merge = True
                for op in subgraphs[i + 1]:
                    for t in inputs[op]:
                        if t in tensor_producer:
                            if (tensor_producer[t] not in candidate
                                    and t not in produced_so_far
                                    and t not in pinned_tensors):
                                can_merge = False
                                break
                    if not can_merge:
                        break

                if can_merge:
                    current = candidate
                    i += 1
                    merged = True
                else:
                    break

            new_subgraphs.append(current)
            i += 1
        subgraphs = new_subgraphs

    # ── Build output ──────────────────────────────────────────────────────────
    result = {"subgraphs": [], "granularities": [], "tensors_to_retain": [],
              "traversal_orders": [], "subgraph_latencies": []}

    # Collect all tensors needed by future subgraphs (for retain decisions)
    def future_needs(step):
        needed = set()
        for sg in subgraphs[step + 1:]:
            for op in sg:
                for t in inputs[op]: needed.add(t)
        return needed

    retained_prev = set(pinned_tensors)  # pinned tensors are retained from the start
    for step, sg in enumerate(subgraphs):
        gran = choose_granularity(sg, retained=retained_prev)

        produced = set()
        for op in sg:
            for t in outputs[op]: produced.add(t)

        # Retain: pinned tensors + produced tensors needed by a future subgraph
        still_needed = future_needs(step)
        retain = list((pinned_tensors & still_needed) | (produced & still_needed))

        latency = calc_latency(problem, sg, gran, retained_prev)

        result["subgraphs"].append(sg)
        result["granularities"].append(gran)
        result["tensors_to_retain"].append(retain)
        result["traversal_orders"].append(None)
        result["subgraph_latencies"].append(round(latency, 2))

        retained_prev = set(retain)

    return result


# ── Simulated annealing post-processor ───────────────────────────────────────

def sa_improve(problem, schedule, time_budget_s):
    """
    Improve a schedule using simulated annealing.
    Tries three moves: split a subgraph, merge two adjacent subgraphs,
    and move a boundary op between adjacent subgraphs.
    Runs for time_budget_s seconds and returns the best schedule found.
    """
    import copy

    widths   = problem["widths"]
    heights  = problem["heights"]
    inputs   = problem["inputs"]
    outputs  = problem["outputs"]
    op_types = problem["op_types"]
    capacity = problem["fast_memory_capacity"]
    native_w, native_h = problem["native_granularity"]
    n_ops = len(op_types)

    # Rebuild pinned tensors
    tensor_producer = {}
    for op in range(n_ops):
        for t in outputs[op]:
            tensor_producer[t] = op
    all_tensors  = set(range(len(widths)))
    graph_inputs = all_tensors - set(tensor_producer.keys())
    tensor_use_count = {}
    for op in range(n_ops):
        for t in inputs[op]:
            if t in graph_inputs:
                tensor_use_count[t] = tensor_use_count.get(t, 0) + 1
    pinned_tensors = {t for t, cnt in tensor_use_count.items() if cnt > 2}
    pinned_size    = sum(widths[t] * heights[t] for t in pinned_tensors)

    def choose_gran(sg_ops, retained=set()):
        produced = set()
        consumed = set()
        for op in sg_ops:
            for t in outputs[op]: produced.add(t)
            for t in inputs[op]:  consumed.add(t)
        ephemeral    = produced & consumed
        boundary_in  = (consumed - produced) - retained
        boundary_out = produced - ephemeral
        has_mm = any(op_types[op] == "MatMul" for op in sg_ops)
        k = native_w if has_mm else 1
        for divisor in [1, 2, 4, 8, 16]:
            w = max(native_w // divisor, 1)
            h = max(native_h // divisor, 1)
            ws = (len(boundary_in) + len(boundary_out)) * w * h
            if has_mm: ws += w * h
            ws += pinned_size
            if ws <= capacity:
                return [w, h, k]
        return [native_w // 16, native_h // 16, native_w]

    def score(sgs):
        total = 0.0
        retained = set(pinned_tensors)
        for idx, sg in enumerate(sgs):
            gran = choose_gran(sg, retained)
            total += calc_latency(problem, sg, gran, retained)
            produced = set()
            for op in sg:
                for t in outputs[op]: produced.add(t)
            future_needed = set()
            for later_sg in sgs[idx + 1:]:
                for op in later_sg:
                    for t in inputs[op]: future_needed.add(t)
            retained = (pinned_tensors & future_needed) | (produced & future_needed)
        return total

    def build_result(sgs):
        result = {"subgraphs": [], "granularities": [], "tensors_to_retain": [],
                  "traversal_orders": [], "subgraph_latencies": []}
        retained = set(pinned_tensors)
        for step, sg in enumerate(sgs):
            gran = choose_gran(sg, retained)
            produced = set()
            for op in sg:
                for t in outputs[op]: produced.add(t)
            future_needed = set()
            for later_sg in sgs[step + 1:]:
                for op in later_sg:
                    for t in inputs[op]: future_needed.add(t)
            retain = list((pinned_tensors & future_needed) | (produced & future_needed))
            latency = calc_latency(problem, sg, gran, retained)
            result["subgraphs"].append(sg)
            result["granularities"].append(gran)
            result["tensors_to_retain"].append(retain)
            result["traversal_orders"].append(None)
            result["subgraph_latencies"].append(round(latency, 2))
            retained = set(retain)
        return result

    def fits_in_memory(sg_ops):
        gran = choose_gran(sg_ops, pinned_tensors)
        produced = set()
        consumed = set()
        for op in sg_ops:
            for t in outputs[op]: produced.add(t)
            for t in inputs[op]:  consumed.add(t)
        ephemeral    = produced & consumed
        boundary_in  = (consumed - produced) - pinned_tensors
        boundary_out = produced - ephemeral
        has_mm = any(op_types[op] == "MatMul" for op in sg_ops)
        w, h, _ = gran
        ws = (len(boundary_in) + len(boundary_out)) * w * h
        if has_mm: ws += w * h
        ws += pinned_size
        return ws <= capacity

    cur_sgs   = [list(sg) for sg in schedule["subgraphs"]]
    cur_cost  = score(cur_sgs)
    best_sgs  = copy.deepcopy(cur_sgs)
    best_cost = cur_cost

    start = time.time()
    t_max = max(cur_cost * 0.1, 1.0)
    t_min = max(cur_cost * 1e-5, 1e-6)
    steps = 0

    # Cache latency weights — recompute only when schedule changes
    cached_latencies = [calc_latency(problem, sg, choose_gran(sg, pinned_tensors), pinned_tensors)
                        for sg in cur_sgs]
    weights_dirty = False

    def weighted_idx(size):
        total = sum(cached_latencies[:size]) or 1.0
        r = random.random() * total
        cumulative = 0.0
        for j in range(size):
            cumulative += cached_latencies[j]
            if r <= cumulative:
                return j
        return size - 1

    while time.time() - start < time_budget_s:
        steps += 1
        elapsed_frac = (time.time() - start) / time_budget_s
        temp = t_max * (t_min / t_max) ** elapsed_frac

        # Refresh weights lazily after an accepted move
        if weights_dirty:
            cached_latencies = [calc_latency(problem, sg, choose_gran(sg, pinned_tensors), pinned_tensors)
                                 for sg in cur_sgs]
            weights_dirty = False

        n = len(cur_sgs)
        if n == 0:
            break
        move = random.randint(0, 2)
        new_sgs = None

        if move == 0 and n >= 2:
            # Merge two adjacent subgraphs — pick the costlier of the pair
            i = min(weighted_idx(n), n - 2)
            merged = cur_sgs[i] + cur_sgs[i + 1]
            if fits_in_memory(merged):
                new_sgs = cur_sgs[:i] + [merged] + cur_sgs[i + 2:]

        elif move == 1:
            # Split the most expensive subgraph at a random point
            i = weighted_idx(n)
            sg = cur_sgs[i]
            if len(sg) >= 2:
                split = random.randint(1, len(sg) - 1)
                new_sgs = cur_sgs[:i] + [sg[:split], sg[split:]] + cur_sgs[i + 1:]

        elif move == 2 and n >= 2:
            # Move last op of subgraph i to the front of subgraph i+1
            i = min(weighted_idx(n), n - 2)
            if len(cur_sgs[i]) >= 2:
                moved_op  = cur_sgs[i][-1]
                new_sg_i  = cur_sgs[i][:-1]
                new_sg_i1 = [moved_op] + cur_sgs[i + 1]
                if fits_in_memory(new_sg_i1):
                    new_sgs = cur_sgs[:i] + [new_sg_i, new_sg_i1] + cur_sgs[i + 2:]

        if new_sgs is None:
            continue

        new_cost = score(new_sgs)
        delta = new_cost - cur_cost
        if delta < 0 or (temp > 0 and random.random() < math.exp(-delta / temp)):
            cur_sgs  = new_sgs
            cur_cost = new_cost
            weights_dirty = True
            if cur_cost < best_cost:
                best_cost = cur_cost
                best_sgs  = copy.deepcopy(cur_sgs)

    print(f"SA: {steps} steps, latency {best_cost:.2f} (was {sum(schedule['subgraph_latencies']):.2f})")
    return build_result(best_sgs)


# ── Make API call to refine scheduler ───────────────────────────
def validate_schedule(problem, schedule):
    """
    Validate LLM output. Returns (ok, reason).
    Checks: all ops present exactly once, no subgraph exceeds fast memory.
    """
    n_ops    = len(problem["op_types"])
    widths   = problem["widths"]
    heights  = problem["heights"]
    inputs   = problem["inputs"]
    outputs  = problem["outputs"]
    op_types = problem["op_types"]
    capacity = problem["fast_memory_capacity"]
    native_w, native_h = problem["native_granularity"]

    subgraphs = schedule.get("subgraphs", [])

    # Check all ops appear exactly once
    all_ops = [op for sg in subgraphs for op in sg]
    if sorted(all_ops) != list(range(n_ops)):
        return False, f"ops mismatch: got {len(all_ops)}, expected {n_ops}"

    # Rebuild pinned tensors
    tensor_producer = {}
    for op in range(n_ops):
        for t in outputs[op]:
            tensor_producer[t] = op
    all_tensors  = set(range(len(widths)))
    graph_inputs = all_tensors - set(tensor_producer.keys())
    use_count = {}
    for op in range(n_ops):
        for t in inputs[op]:
            if t in graph_inputs:
                use_count[t] = use_count.get(t, 0) + 1
    pinned = {t for t, c in use_count.items() if c > 2}
    pinned_size = sum(widths[t] * heights[t] for t in pinned)

    # Check each subgraph fits in memory
    for i, sg in enumerate(subgraphs):
        produced = set()
        consumed = set()
        for op in sg:
            for t in outputs[op]: produced.add(t)
            for t in inputs[op]:  consumed.add(t)
        ephemeral    = produced & consumed
        boundary_in  = (consumed - produced) - pinned
        boundary_out = produced - ephemeral
        ws = (len(boundary_in) + len(boundary_out)) * native_w * native_h + pinned_size
        if ws > capacity:
            return False, f"subgraph {i} OOM: working_set={ws} > capacity={capacity}"

    return True, "ok"

def llm_refine(client, problem, baseline_schedule, sys_instr, start_time, timeout):
    """
    Make a SINGLE API call asking the LLM to improve the baseline schedule.
    Retries on rate limits only while there is time remaining.
    Validates output before accepting — falls back to None on invalid output.
    """
    baseline_latency = sum(baseline_schedule.get("subgraph_latencies", [0]))

    # Compute per-subgraph working set info to warn LLM about memory constraints
    n_ops   = len(problem["op_types"])
    widths  = problem["widths"]
    heights = problem["heights"]
    inp     = problem["inputs"]
    outp    = problem["outputs"]
    capacity = problem["fast_memory_capacity"]
    native_w, native_h = problem["native_granularity"]
    tensor_producer = {}
    for op in range(n_ops):
        for t in outp[op]: tensor_producer[t] = op
    graph_inputs = set(range(len(widths))) - set(tensor_producer.keys())
    use_count = {}
    for op in range(n_ops):
        for t in inp[op]:
            if t in graph_inputs: use_count[t] = use_count.get(t, 0) + 1
    pinned = {t for t, c in use_count.items() if c > 2}
    pinned_size = sum(widths[t] * heights[t] for t in pinned)

    def working_set(sg_ops):
        produced, consumed = set(), set()
        for op in sg_ops:
            for t in outp[op]: produced.add(t)
            for t in inp[op]:  consumed.add(t)
        ephemeral = produced & consumed
        bi = (consumed - produced) - pinned
        bo = produced - ephemeral
        return (len(bi) + len(bo)) * native_w * native_h + pinned_size

    prompt = (
        "You are given a hardware scheduling problem and a baseline schedule produced by simulated annealing.\n"
        f"The baseline total latency is {baseline_latency:.2f}. Your goal is to produce a schedule with LOWER total latency.\n\n"
        f"MEMORY CONSTRAINT: fast_memory_capacity={capacity}, pinned_size={pinned_size} (always occupied).\n"
        f"Each subgraph's working_set = (boundary_inputs + boundary_outputs) * {native_w} * {native_h} + {pinned_size} must be <= {capacity}.\n\n"
        "RULES:\n"
        "1. Every op must appear exactly once across all subgraphs.\n"
        "2. No subgraph may exceed the memory capacity (check working_set before grouping).\n"
        "3. Do NOT split subgraphs that are already merged — merging reduces latency by eliminating memory round-trips.\n"
        "4. Only regroup ops if you are confident the new grouping fits in memory AND reduces total latency.\n"
        "5. If you cannot improve the schedule, return it unchanged.\n\n"
        "Return ONLY valid JSON matching the output schema.\n\n"
        f"PROBLEM:\n{json.dumps(problem)}\n\n"
        f"BASELINE SCHEDULE (total_latency={baseline_latency:.2f}, do not make this worse):\n{json.dumps(baseline_schedule)}"
    )

    attempt = 0
    while True:
        elapsed = time.time() - start_time
        remaining = timeout - elapsed
        if remaining < 30:
            print(f"Skipping LLM call — only {remaining:.0f}s remaining.")
            return None

        try:
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=sys_instr,
                    response_mime_type="application/json",
                    temperature=0.0,
                    max_output_tokens=65536
                )
            )
            res_text = response.text if response.text else "{}"
            refined = json.loads(res_text)
            if isinstance(refined, dict) and "subgraphs" in refined:
                refined.pop("scratchpad", None)
                ok, reason = validate_schedule(problem, refined)
                if ok:
                    refined_latency = sum(refined.get("subgraph_latencies", [0]))
                    if refined_latency >= baseline_latency * 1.01:  # allow 1% tolerance
                        print(f"LLM output worse than baseline ({refined_latency:.2f} >= {baseline_latency:.2f}), retrying...")
                        attempt += 1
                        continue
                    return refined
                else:
                    print(f"LLM output invalid ({reason}), retrying...")
                    attempt += 1
                    continue
        except Exception as e:
            if "429" in str(e):
                if attempt >= 3:
                    print("Rate limited — max retries reached, using SA schedule.")
                    return None
                wait = min(30 * (attempt + 1) + random.uniform(0, 5), remaining - 30)
                if wait <= 0:
                    print("Rate limited but no time left to retry.")
                    return None
                print(f"Rate limit hit. Waiting {wait:.1f}s... ({remaining:.0f}s remaining)")
                time.sleep(wait)
                attempt += 1
            else:
                print(f"LLM refinement error: {e}, retrying...")
                attempt += 1
                continue

def main(input_path, output_path):
    start_time = time.time()
    TIMEOUT = 540  # 9 minutes — leave 1 min buffer before the 10-min hard limit

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("Error: GOOGLE_API_KEY not found.")
        sys.exit(1)

    client = genai.Client(api_key=api_key)

    with open(input_path, 'r') as f:
        problem = json.load(f)
    try:
        with open("prompts/system_prompt.txt", "r") as f:
            sys_instr = f.read()
    except FileNotFoundError:
        sys_instr = "You are an ML hardware scheduling agent. Minimize total latency."

    print("Building baseline schedule (no API call)...")
    baseline = greedy_schedule(problem)
    baseline_latency = sum(baseline["subgraph_latencies"])
    print(f"Baseline total latency: {baseline_latency:.2f} ({len(baseline['subgraphs'])} subgraphs)")

    # SA gets 3 minutes, leaving ~6 min for the LLM call + retries
    sa_budget = min(180, TIMEOUT - (time.time() - start_time) - 120)
    if sa_budget > 5:
        print(f"Running simulated annealing ({sa_budget:.0f}s budget)...")
        improved = sa_improve(problem, baseline, sa_budget)
    else:
        improved = baseline

    print("Requesting LLM refinement (1 API call)...")
    refined = llm_refine(client, problem, improved, sys_instr, start_time, TIMEOUT)

    if refined:
        final = refined
        print("LLM refinement accepted.")
    else:
        final = improved
        print("Using SA-improved schedule (LLM refinement failed).")

    with open(output_path, 'w') as f:
        json.dump(final, f, indent=2)

    elapsed = time.time() - start_time
    print(f"Done in {elapsed/60:.1f}m — schedule saved to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python agent.py <input> <output>")
    else:
        main(sys.argv[1], sys.argv[2])
