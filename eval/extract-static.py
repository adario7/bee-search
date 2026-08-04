import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arena import Engine
import argparse, json, os, re
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

def parse_int_from_resp(resp):
    # resp is a list of strings; find first full integer or fallback to first integer substring
    for s in resp:
        s = s.strip()
        if re.fullmatch(r"-?\d+", s):
            return int(s)
    joined = " ".join(resp)
    nums = re.findall(r"-?\d+", joined)
    if nums:
        return int(nums[0])
    raise ValueError("No integer found in response")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--engine", default="build/release/bee-search")
    p.add_argument("--evaluations-path", default="logs/evaluations.json")
    p.add_argument("--out-path", default="logs/position_static.json")
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
            engine.send("static_eval")
            resp = engine.receive(timeout=args.timeout)
            if not resp:
                raise TimeoutError("no static_eval response")
            val = parse_int_from_resp(resp)
            results.append({"engine": engine.name, "position": pos, "static_eval": int(val)})
        except Exception as e:
            try:
                engine.terminate()
            except Exception:
                pass
            engine = init_engine(args.engine)
            # one retry
            try:
                engine.send(f"newgame {pos}"); engine.receive(timeout=1)
                engine.send("static_eval")
                resp = engine.receive(timeout=args.timeout)
                val = parse_int_from_resp(resp) if resp else None
                if val is not None:
                    results.append({"engine": engine.name, "position": pos, "static_eval": int(val)})
            except Exception:
                print("SKIP", pos, e)

    os.makedirs(os.path.dirname(args.out_path) or ".", exist_ok=True)
    with open(args.out_path, "w") as f:
        json.dump(results, f, indent=1)
    print(f"Saved static evals for {len(results)} positions to {args.out_path}")
