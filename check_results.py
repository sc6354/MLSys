import json
import sys

def verify_completeness(problem_path, solution_path):
    with open(problem_path, 'r') as f:
        problem = json.load(f)
    with open(solution_path, 'r') as f:
        solution = json.load(f)
        
    expected_ops = set(range(len(problem['op_types'])))
    actual_ops = set()
    for sub in solution['subgraphs']:
        actual_ops.update(sub)
        
    missing = expected_ops - actual_ops
    extra = actual_ops - expected_ops
    
    if not missing and not extra:
        print(f"✅ {solution_path}: All {len(expected_ops)} operations accounted for.")
    else:
        print(f"❌ {solution_path}: Missing {len(missing)} ops, Extra {len(extra)} ops.")

if __name__ == "__main__":
    verify_completeness(sys.argv[1], sys.argv[2])