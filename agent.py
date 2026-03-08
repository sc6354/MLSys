iimport sys, json, os, time
from google import genai
from google.genai import types

def main(input_path, output_path):
    start_time = time.time()
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key: sys.exit(1)
    
    client = genai.Client(api_key=api_key)

    with open(input_path, 'r') as f:
        problem_data = json.load(f)

    # 1. Load System Prompt from the required prompts/ folder
    with open("prompts/system_prompt.txt", "r") as f:
        sys_instr = f.read()

    # 2. Chunking Logic (Chunk size 40 to stay under 1,500 RPD quota)
    all_ops = list(range(len(problem_data['op_types'])))
    chunk_size = 40 
    chunks = [all_ops[i:i + chunk_size] for i in range(0, len(all_ops), chunk_size)]
    
    final_data = {"subgraphs": [], "granularities": [], "tensors_to_retain": [], "traversal_orders": [], "subgraph_latencies": []}
    active_in_cache = []

    for i, chunk_indices in enumerate(chunks):
        # 3. Check for 10-minute timeout (600 seconds)
        if time.time() - start_time > 540: # 9-minute buffer
            print("⏳ Approaching contest timeout! Saving partial results.")
            break

        mini_problem = {
            "op_types": [problem_data['op_types'][j] for j in chunk_indices],
            "inputs": [problem_data['inputs'][j] for j in chunk_indices],
            "outputs": [problem_data['outputs'][j] for j in chunk_indices],
            "currently_in_cache": active_in_cache
        }
        
        prompt = f"Schedule CHUNK {i}. Tensors in cache: {active_in_cache}.\n{json.dumps(mini_problem)}"
        
        try:
            # 4. Use the specific model mentioned in contest rules
            response = client.models.generate(
                model='gemini-1.5-flash', 
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=sys_instr,
                    response_mime_type="application/json",
                    temperature=0.0
                )
            )
            
            chunk_result = json.loads(response.text)
            for sub in chunk_result.get('subgraphs', []):
                final_data["subgraphs"].append([chunk_indices[op_id] for op_id in sub])
            
            final_data["granularities"].extend(chunk_result.get('granularities', []))
            final_data["tensors_to_retain"].extend(chunk_result.get('tensors_to_retain', []))
            final_data["traversal_orders"].extend(chunk_result.get('traversal_orders', [None]*len(chunk_result.get('subgraphs', []))))
            final_data["subgraph_latencies"].extend(chunk_result.get('subgraph_latencies', [0.0]*len(chunk_result.get('subgraphs', []))))
            
            if chunk_result.get('tensors_to_retain'):
                active_in_cache = chunk_result['tensors_to_retain'][-1]
        except Exception:
            continue

    with open(output_path, 'w') as f:
        json.dump(final_data, f, indent=2)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])