from arena import Engine
import os
import json
from tqdm import tqdm

default_engine_path = "build/debug/bee-search.exe"

def get_engine_eval(engine_path, positions, depth=5, timeout = 120):

    engine = Engine(engine_path)

    engine.send("newgame Base")
    result = engine.receive(timeout=1)
    if len(result) == 0:
        raise TimeoutError("Engine didn't respond")
    engine.send("bestmove depth 1")
    result = engine.receive(timeout=1)
    if len(result) == 0:
        raise TimeoutError("Engine didn't respond")

    scores = []
    
    for position in tqdm(positions):
        try:
            engine.send(f"newgame {position}")
            result = engine.receive(timeout=1)
            if len(result) == 0:
                raise TimeoutError("Engine didn't respond")
            
            engine.send(f"eval {depth}")
            result = engine.receive(timeout=timeout)
            
            if len(result) != 0:
                result = result[0]

                if len(result) < 10:
                    pos = {}
                    pos["engine"] = engine.name
                    pos["depth"] = depth
                    pos["position"] = position
                    pos["evaluation"] = int(result)
                    scores.append(pos)
                else:
                    print(position)
                    print(result)
        except Exception as e:
            print(position)
            print(e)
            engine.terminate()
            engine = Engine(engine_path)
            engine.send("newgame Base")
            result = engine.receive(timeout=1)
            if len(result) == 0:
                raise TimeoutError("Engine didn't respond")
            engine.send("bestmove depth 1")
            result = engine.receive(timeout=1)
            if len(result) == 0:
                raise TimeoutError("Engine didn't respond")

    return scores 

def load_results(results_file):
    if os.path.exists(results_file):
        with open(results_file, "r") as f:
            return json.load(f)
    else:
        return []

def get_positions_from_results(results_paths, engine = Engine(default_engine_path)):
    positions = []

    for path in results_paths:
        results = load_results(path)

        for result in results:
            tmp = result["final_gamestate"].split(';')
            
            gametype = tmp[0]
            moves = tmp[3:]

            engine.send(f"newgame {gametype}")
            position = engine.receive(timeout=1)
            if len(position) == 0:
                raise TimeoutError("Engine didn't respond")
            
            for move in moves:
                engine.send(f"play {move}")
                position = engine.receive(timeout=1)
                if len(position) == 0:
                    raise TimeoutError("Engine didn't respond")
                
                positions.append(position[0])

    return set(positions)

if __name__ == "__main__":
    results_paths = ["logs/results.json", "logs/500ms/resutls.json", "logs/1000ms/results.json"]

    positions = list(get_positions_from_results(results_paths=results_paths))

    print(f"Found {len(positions)} different positions")

    evals = get_engine_eval(engine_path=default_engine_path, positions=positions, depth=2)

    with open("logs/evaluations.json", "w") as f:
        json.dump(evals, f, indent=2)
