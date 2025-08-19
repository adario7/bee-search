from arena import Engine
import os
import json
from tqdm import tqdm
import numpy as np
import pickle
from arena import game_is_ended

default_engine_path = "build/release/bee-search"
FEATURES_LEN = 12

def get_engine_eval(engine_path, positions, depth=5, timeout=120):
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

                    engine.send("static_eval")
                    result = engine.receive(timeout=1)
                    if len(result) > 0 and (result[0].isdigit() or (result[0][0] == "-" and result[0][1:].isdigit())):
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
    engine.terminate()
    return scores

def get_fair_starting_positions(engine_path, random_moves=4, n_runs = 1000, depth = 8):
    engine = Engine(engine_path)
    starting_position = "Base+MLP;NotStarted;White[1]"
    valid_positions = set()

    for _ in range(n_runs):
        position = starting_position
        n_moves = 0
        while n_moves < random_moves:
            if game_is_ended(position):
                break
            try:
                engine.send(f"newgame {position}")
                response = engine.receive(timeout=1)
                if len(response) == 0:
                    raise TimeoutError(f"No newgame response from engine.")

                engine.send(f"validmoves")
                validmoves = engine.receive(timeout=1)
                if len(validmoves)==0:
                    raise TimeoutError(f"Engine {engine.name} couldn't find valid moves")
                validmoves = validmoves[0].split(';')

                move = str(np.random.choice(validmoves))
                    
                engine.send(f"play {move}")
                response = engine.receive(timeout=1)
                if len(response)==0:
                    raise TimeoutError(f"No play from engine {engine.name}.")
                prev_position = position
                position = response[0]
                n_moves += 1
            except TimeoutError as e:
                error = str(e)
                print(e)
                print(f"Unable to connnect with engine {engine.name}")
                break
            except IndexError as e:
                error = str(e)
                print(e)
                print("Probably an invalid move was made in this position:")
                print(prev_position)
                position = prev_position
                break
                
        if n_moves == random_moves:
            valid_positions.add(position)
        
    return get_engine_eval(engine_path=engine_path, positions=list(valid_positions), depth=depth)

def load_results(results_file):
    if not os.path.exists(results_file):
        print(f"W: file {results_file} not found")
        return []
    with open(results_file, "r") as f:
        return json.load(f)

def get_positions_from_results(results_paths, engine):
    positions = set()
    winners = {}
    opp = {"white": "black", "black": "white"}

    for path in results_paths:
        print(f"Extracting positions from results in {path}...")
        results = load_results(path)

        for result in tqdm(results):
            tmp = result["final_gamestate"].split(";")
            winner = result["winner"]

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
                position = position[0]
                if "invalidmove" in position:
                    break
                positions.add(position)
                turn = position.split(";")[2].split("[")[0].lower()
                relative = 1 if winner==turn else 0 if winner==opp[turn] else 0.5
                if position not in winners: winners[position] = { 0: 0, 0.5: 0, 1: 0 }
                winners[position][relative] += 1

    positions = list(positions)
    print(f"Extracted {len(positions)} positions.")

    win_prob = {}
    for position in positions:
        wp, tot = 0, 0
        for (w, n) in winners[position].items():
            wp += w * n
            tot += n
        win_prob[position] = wp / tot

    return positions, win_prob

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
            features[i] = np.array([float(x) for x in result[up_to + i].split(" ")])

        up_to += n
        edges = np.zeros((n, n), dtype=bool)
        for i in range(n):
            edges[i] = np.array([int(x) for x in result[up_to + i].split(" ")])

        up_to += n
        pieces_on_tile = []
        for i in range(n):
            height = int(result[up_to])
            up_to += 1
            if height == 0:
                pieces_on_tile.append(None)
                continue

            height -= 1
            top_piece = [int(x) for x in result[up_to].split(" ")]
            up_to += 1
            pieces = [[int(x) for x in result[j].split(" ")] for j in range(up_to, up_to + height)]
            pieces = [top_piece] + pieces[::-1]
            pieces_on_tile.append(pieces)

            up_to += height

        graphs.append({"position": position, "graph": [nodes, features, edges], "pieces_on_tile": pieces_on_tile})

    return graphs

def get_global_features_from_positions(positions, engine):
    global_features = []

    print("Generating global features...")

    for position in tqdm(positions):
        engine.send(f"newgame {position}")
        result = engine.receive(timeout=1)
        if len(result) == 0:
            raise TimeoutError("Engine didn't respond")

        engine.send("global_features")
        result = engine.receive(timeout=1)
        if len(result) == 0:
            raise TimeoutError("Engine didn't respond")

        features = {}

        for line in result:
            key, value = line.split(": ", 1)
            match key:
                case "queen_score":
                    features["queen_score"] = int(value)
                case "other_queen_score":
                    features["other_queen_score"] = int(value)
                case "n_moves":
                    features["n_moves"] = int(value)
                case "other_n_moves":
                    features["other_n_moves"] = int(value)
                case "tiles_placed":
                    tiles_placed = [int(x) for x in value.split()]
                    features["tiles_placed"] = tiles_placed
                case "other_tiles_placed":
                    other_tiles_placed = [int(x) for x in value.split()]
                    features["other_tiles_placed"] = other_tiles_placed
                case _:
                    raise ValueError(f"Unknown feature: {key} with value {value}")

        global_features.append(features)

    return global_features

def plan_evaluation_run(all_positions, evals_path, depth, engine_name, reevaluate):
    existing_evals = load_results(evals_path)
    evals_map = {e["position"]: e for e in existing_evals}

    positions_to_eval = []
    num_present_diff_params = 0
    num_missing = 0

    for pos in all_positions:
        if pos in evals_map:
            existing_eval = evals_map[pos]
            is_different = (existing_eval.get("depth") != depth or existing_eval.get("engine") != engine_name)
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
    parser.add_argument("--results-paths", type=str, nargs="+", default=["logs/results.json"],
                        help="List of result files.")
    parser.add_argument("--engine", type=str, default=default_engine_path,
                        help="Path to the Hive engine executable.")
    parser.add_argument("--depth", type=int, default=5,
                        help="Depth for engine evaluation.")
    parser.add_argument("--timeout", type=int, default=10,
                        help="Timeout for engine responses in seconds.")
    parser.add_argument("--evals-path", type=str, default="logs/evaluations.json",
                        help="Path to save the evaluations JSON file.")
    parser.add_argument("--reevaluate", action="store_true",
                        help="Re-evaluate positions if depth or engine name are different from the one we are using now.")
    parser.add_argument("--global-features-path", type=str, default=None,
                        help="Path to save the global features JSON file.")
    parser.add_argument("--graphs-path", type=str, default=None,
                        help="Path to save the graphs pickle file.")
    parser.add_argument("--get-fair-positions", action="store_true")

    args = parser.parse_args()

    if args.get_fair_positions:
        scores = get_fair_starting_positions(engine_path=args.engine)
        with open("logs/fair_positions.json", "w") as f:
            json.dump(scores, f, indent=2)
        exit(0) 

    engine = Engine(args.engine)
    positions, win_prob = get_positions_from_results(args.results_paths, engine)

    current_engine_name = engine.name

    # Plan the evaluation run
    positions_to_eval, evals_map = plan_evaluation_run(positions, args.evals_path, args.depth, current_engine_name, args.reevaluate)

    # update winners
    for pos in positions:
        if pos in evals_map:
            evals_map[pos]["winner"] = win_prob[pos]

    # Run evaluation on the filtered list
    if positions_to_eval:
        print(f"Evaluating {len(positions_to_eval)} positions...")
        chunk_size = 1000
        processed = 0
        for i in range(0, len(positions_to_eval), chunk_size):
            chunk = positions_to_eval[i:i + chunk_size]
            evals = get_engine_eval(engine_path=args.engine, positions=chunk, depth=args.depth, timeout=args.timeout)
            for e in evals:
                evals_map[e["position"]] = e
                e["winner"] = win_prob[e["position"]]
            processed += len(chunk)
            # Save after each chunk so we don't lose progress (including re-evaluations)
            with open(args.evals_path, "w") as f:
                json.dump(list(evals_map.values()), f, indent=1)
            print(f"Saved {len(evals_map)} evaluations to {args.evals_path} after processing {processed} positions")
    else:
        print("No new positions to evaluate.")

    # Save the consolidated list of all evaluations (old and new)
    with open(args.evals_path, "w") as f:
        json.dump(list(evals_map.values()), f, indent=1)
    print(f"Evaluations saved to {args.evals_path}")

    if args.global_features_path:
        global_features = get_global_features_from_positions(positions=positions, engine=engine)
        with open(args.global_features_path, "w") as f:
            json.dump(global_features, f, indent=1)

    if args.graphs_path:
        graphs = get_graph_from_positions(positions=positions, engine=engine)
        with open(args.graphs_path, "wb") as f:
            pickle.dump(graphs, f)
    
    engine.terminate()
