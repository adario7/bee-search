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
        e.send("newgame Base"); e.receive(timeout=timeout)
        e.send("bestmove depth 1"); e.receive(timeout=timeout)
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
