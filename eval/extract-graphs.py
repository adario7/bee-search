import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arena import Engine
import argparse, json, os
from tqdm import tqdm

def load_json(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "r") as f:
        if path.endswith(".jsonl"):
            return [json.loads(line) for line in f if line.strip()]
        return json.load(f)

def init_engine(path, timeout=1):
    e = Engine(path)
    try:
        e.command("newgame Base", timeout)
    except Exception:
        pass
    return e

def parse_graph_resp(resp):
    num_nodes = int(resp[0].strip())
    features = []
    for i in range(1, num_nodes + 1):
        vec = [int(x) for x in resp[i].strip().split() if x != ""]
        features.append(vec)
    
    num_edges_idx = num_nodes + 1
    num_edges = int(resp[num_edges_idx].strip())
    edges_src = []
    edges_dst = []
    edges_cat = []
    for i in range(num_edges_idx + 1, num_edges_idx + 1 + num_edges):
        u, v, c = [int(x) for x in resp[i].strip().split() if x != ""]
        edges_src.append(u)
        edges_dst.append(v)
        edges_cat.append(c)
        
    return {
        "nodes": features,
        "edges": [edges_src, edges_dst, edges_cat]
    }

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--engine", default="build/release/bee-search")
    p.add_argument("--evaluations-path", default="logs/evaluations.json")
    p.add_argument("--out-path", default="logs/position_graphs.jsonl")
    p.add_argument("--timeout", type=int, default=5)
    args = p.parse_args()

    evals = load_json(args.evaluations_path)
    engine = init_engine(args.engine)

    os.makedirs(os.path.dirname(args.out_path) or ".", exist_ok=True)
    count = 0
    with open(args.out_path, "w") as f:
        for rec in tqdm(evals):
            pos = rec.get("position")
            if not pos:
                continue
                
            for attempt in range(2):
                try:
                    engine.command(f"newgame {pos}", timeout=1)
                    
                    resp = engine.command("graph", timeout=args.timeout)
                    if not resp:
                        raise TimeoutError("no graph response")
                    
                    graph_data = parse_graph_resp(resp)
                    res = {"engine": engine.name, "position": pos, "graph": graph_data}
                    f.write(json.dumps(res) + "\n")
                    count += 1
                    break
                    
                except Exception as e:
                    if attempt == 0:
                        try:
                            engine.terminate()
                        except Exception:
                            pass
                        engine = init_engine(args.engine)
                    else:
                        print(f"SKIP {pos}: {e}")

    print(f"Saved graphs for {count} positions to {args.out_path}")
