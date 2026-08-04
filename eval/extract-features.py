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
        return json.load(f)

def init_engine(path, timeout=1):
    e = Engine(path)
    try:
        e.command("newgame Base", timeout)
    except Exception:
        pass
    return e

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--engine", default="build/release/bee-search")
    p.add_argument("--evaluations-path", default="logs/evaluations.json")
    p.add_argument("--out-path", default="logs/position_features.json")
    p.add_argument("--timeout", type=int, default=5)
    args = p.parse_args()

    evals = load_json(args.evaluations_path)
    engine = init_engine(args.engine)

    results = []
    for rec in tqdm(evals):
        pos = rec.get("position")
        if not pos:
            continue
        try:
            engine.send(f"newgame {pos}")
            resp = engine.receive(timeout=1)
            if not resp:
                raise TimeoutError("no response to newgame")
            engine.send("features")
            resp = engine.receive(timeout=args.timeout)
            if not resp:
                raise TimeoutError("no features response")
            line = ";".join(resp).strip()
            vec = [int(x) for x in line.split(";") if x != ""]
            results.append({"engine": engine.name, "position": pos, "features": vec})
        except Exception as e:
            try:
                engine.terminate()
            except:
                pass
            engine = init_engine(args.engine)
            # one retry
            try:
                engine.send(f"newgame {pos}"); engine.receive(timeout=1)
                engine.send("features")
                resp = engine.receive(timeout=args.timeout)
                line = ";".join(resp).strip() if resp else ""
                vec = [int(x) for x in line.split(";") if x != ""]
                results.append({"engine": engine.name, "position": pos, "features": vec})
            except Exception:
                print("SKIP", pos, e)

    os.makedirs(os.path.dirname(args.out_path) or ".", exist_ok=True)
    with open(args.out_path, "w") as f:
        json.dump(results, f, indent=1)
    print(f"Saved features for {len(results)} positions to {args.out_path}")

    if results:
        total_zeros = sum(f == 0 for rec in results for f in rec["features"])
        total_elems = sum(len(rec["features"]) for rec in results)
        pct_zero = (total_zeros / total_elems) * 100.0 if total_elems > 0 else 0.0
        print(f"Average percentage of zero entries: {pct_zero:.2f}%")

