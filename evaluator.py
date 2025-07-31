from arena import Engine
import os
import json
from tqdm import tqdm
import numpy as np

default_engine_path = "build/debug/bee-search"
FEATURES_LEN = 12

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

def get_graph_from_positions(positions, engine = Engine(default_engine_path)):

    graphs = []

    print("Generating graphs...")

    for position in tqdm(positions):
        engine.send(f"newgame {position}")
        result = engine.receive(timeout=1)
        if len(result) == 0:
            raise TimeoutError("Engine didn't respond")

        engine.send("graph")
        result = engine.receive(timeout=1)
        if len(result) == 0:
            raise TimeoutError("Engine didn't respond")

        n = int(result[0])
        nodes = np.zeros(n, dtype=np.int16)

        up_to = 1
        for i in range(n):
            nodes[i] = int(result[i + up_to])

        features = np.zeros((n, FEATURES_LEN), dtype=np.float32)
        up_to += n
        for i in range(n):
            features[i] = np.array([float(x) for x in result[up_to + i].split(' ')])

        up_to += n
        edges = np.zeros((n, n), dtype=bool)
        for i in range(n):
            edges[i] = np.array([int(x) for x in result[up_to + i].split(' ')])

        graphs.append({"position": position, "graph": [nodes, features, edges]})

    return graphs

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate positions using the Hive engine.")
    parser.add_argument('--results-paths', type=str, nargs='+', default=['logs/results.json', 'logs/500ms/results.json', 'logs/1000ms/results.json'],
                        help='List of result files.')
    parser.add_argument('--engine-path', type=str, default=default_engine_path,
                        help='Path to the Hive engine executable.')
    parser.add_argument('--depth', type=int, default=5,
                        help='Depth for engine evaluation.')
    parser.add_argument('--timeout', type=int, default=120,
                        help='Timeout for engine responses in seconds.')
    parser.add_argument('--evals-path', type=str, default='logs/evaluations.json',
                        help='Path to save the evaluations JSON file.')

    args = parser.parse_args()

    positions = list(get_positions_from_results(args.results_paths, engine=Engine(args.engine_path)))
    evals = get_engine_eval(engine_path=args.engine_path, positions=positions, depth=args.depth, timeout=args.timeout)
    with open(args.evals_path, "w") as f:
        json.dump(evals, f, indent=2)
    print(f"Evaluations saved to {args.evals_path}")

    graphs = get_graph_from_positions(positions=positions, engine=Engine(args.engine_path))
    with open("logs/graphs.pkl", "wb") as f:
        pickle.dump(graphs, f)
