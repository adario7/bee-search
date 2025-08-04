from arena import Engine
import os
import json
from tqdm import tqdm
import numpy as np
import pickle

default_engine_path = "build/release/bee-search"
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

                    engine.send('static_eval')
                    result = engine.receive(timeout=1)
                    if len(result) > 0 and (result[0].isdigit() or (result[0][0] == '-' and result[0][1:].isdigit())):
                        pos["static_eval"] = int(result[0])

                    scores.append(pos)
                else:
                    raise ValueError(f"Unexpected result format: {result}")

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
    if not os.path.exists(results_file):
        print(f"W: file {results_file} not found")
        return []
    with open(results_file, "r") as f:
        return json.load(f)

def get_positions_from_results(results_paths, engine):

    positions = []

    for path in results_paths:
        print(f"Extracting positions from results in {path}...")
        results = load_results(path)

        for result in tqdm(results):
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

    positions = set(positions)
    print(f"Extracted {len(positions)} positions.")

    return positions

def get_graph_from_positions(positions, engine):

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

def plan_evaluation_run(all_positions, evals_path, depth, engine_name, reevaluate):
    existing_evals = load_results(evals_path)
    evals_map = {e['position']: e for e in existing_evals}

    positions_to_eval = []
    num_present_diff_params = 0
    num_missing = 0
    
    for pos in all_positions:
        if pos in evals_map:
            existing_eval = evals_map[pos]
            is_different = (existing_eval.get('depth') != depth or
                            existing_eval.get('engine') != engine_name)
            if is_different:
                num_present_diff_params += 1
                if reevaluate:
                    positions_to_eval.append(pos)
        else:
            num_missing += 1
            positions_to_eval.append(pos)

    # Print a small summary
    print(f"Num loaded evals: {len(evals_map)}")
    print(f"Present but with different params: {num_present_diff_params}")
    print(f"Missing ones: {num_missing}")

    return positions_to_eval, evals_map


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate positions using the Hive engine.")
    parser.add_argument('--results-paths', type=str, nargs='+', default=['logs/results.json'],
                        help='List of result files.')
    parser.add_argument('--engine', type=str, default=default_engine_path,
                        help='Path to the Hive engine executable.')
    parser.add_argument('--depth', type=int, default=5,
                        help='Depth for engine evaluation.')
    parser.add_argument('--timeout', type=int, default=10,
                        help='Timeout for engine responses in seconds.')
    parser.add_argument('--evals-path', type=str, default='logs/evaluations.json',
                        help='Path to save the evaluations JSON file.')
    parser.add_argument('--reevaluate', action='store_true',
                        help='Re-evaluate positions if depth or engine name are different from the one we are using now.')

    args = parser.parse_args()

    engine = Engine(args.engine)
    positions = list(get_positions_from_results(args.results_paths, engine))
    
    current_engine_name = engine.name
   
    # Plan the evaluation run
    positions_to_eval, evals_map = plan_evaluation_run(positions, args.evals_path, args.depth, current_engine_name, args.reevaluate)

    # Run evaluation on the filtered list
    if positions_to_eval:
        print(f"Evaluating {len(positions_to_eval)} positions...")
        evals = get_engine_eval(engine_path=args.engine, positions=positions_to_eval, depth=args.depth, timeout=args.timeout)
        # Update the main map with new/updated results
        for e in evals:
            evals_map[e['position']] = e
    else:
        print("No new positions to evaluate.")

    # Save the consolidated list of all evaluations (old and new)
    with open(args.evals_path, "w") as f:
        json.dump(list(evals_map.values()), f, indent=1)
    print(f"Evaluations saved to {args.evals_path}")

    graphs = get_graph_from_positions(positions=positions, engine=engine)
    with open("logs/graphs.pkl", "wb") as f:
        pickle.dump(graphs, f)
