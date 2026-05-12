import subprocess
import threading
import queue
import time
import datetime
import json
import os
import argparse
import random
import numpy as np
progress_bar = True
try:
    from tqdm import tqdm
except:
    progress_bar = False

MAX_THINK_TIME = 120  # maximum time for the engine to think (used if depth is used instead of timeout)
THINK_TOL = 0.1  # tolerance for timeout
maxmoves = 200 # maximum number of moves per player per game

class Engine:
    def __init__(self, path, name=None):
        self.path = path
        self.proc = subprocess.Popen(
            [path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._stdout_queue = queue.Queue()
        threading.Thread(target=self._reader, daemon=True).start()

        info = self.receive()
        self.name = name or info[0].split('id ')[1] # the name is the id written at the start of the engine
                                                    # it should be different for each engine 
        self._stderr_queue = queue.Queue()
        self.stderr_log_file = open(f"logs/{self.name}_stderr.log","w")
        threading.Thread(target=self._read_stderr, daemon=True).start()

        # nokamute format
        self.command("options set NumThreads 1")
        self.command("options set TableSizeMiB 256") # max value
        # mzinga format
        self.command("options set MaxHelperThreads None")
        self.command("options set TranspositionTableSizeMB 1024") # max value

    def _reader(self):
        for line in self.proc.stdout:
            self._stdout_queue.put(line)

    def _read_stderr(self):
        for line in self.proc.stderr:
            self._stderr_queue.put(line)
            self.stderr_log_file.write(line)
            self.stderr_log_file.flush()

    def send(self, text):
        if self.proc.stdin:
            self.proc.stdin.write(text + "\n")
            self.proc.stdin.flush()

    def command(self, text, timeout=None):
        self.send(text)
        return self.receive(timeout=timeout)

    def receive(self, timeout=None):
        lines = []
        start_time = time.time()
        while True:
            remaining = None if timeout is None else max(0, timeout - (time.time() - start_time))
            try:
                line = self._stdout_queue.get(timeout=remaining).strip()
                if line == "ok":
                    break
                lines.append(line)
            except queue.Empty:
                break
        return lines

    def terminate(self):
        self.proc.terminate()
        try:
            self.proc.wait(timeout=1)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        self.stderr_log_file.close()

    def restart(self):
        print(f"Try restarting the engine [{self.name}]")
        self.terminate()
        self.proc = subprocess.Popen(
            [self.path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._stdout_queue = queue.Queue()
        threading.Thread(target=self._reader, daemon=True).start()


def seconds_to_hh(seconds):
    time = int(seconds)
    s = time%60
    time //= 60
    m = time%60
    h = time//60

    return f"{h}:{m}:{s}"

def get_winner_from_gamestate(position):
    status = position.split(';')[1]

    if status == 'WhiteWins':
        return 'white'
    if status == 'BlackWins':
        return 'black'
    if status == 'Draw':
        return 'draw'
    return 'other'

def game_is_ended(position):
    status = position.split(';')
    if len(status) > 1:
        status = status[1]
    else:
        raise IndexError("Status has lenght 1")


    if status != 'NotStarted' and status != 'InProgress':
        return True
    return False

def path_to_name(path):
    engine = Engine(path)
    name = engine.name
    engine.terminate()
    return name

class HiveArena:
    def __init__(self, engine_paths, results_folder="logs/"):
        self.engine_paths = engine_paths
        self.results_file = os.path.join(results_folder , "results.json")
        self.ratings_file = os.path.join(results_folder, "ratings.json")
        self.starting_position = "Base+MLP;NotStarted;White[1]"

        self.load_results()
        self.load_ratings()

    def load_results(self):
        if os.path.exists(self.results_file):
            with open(self.results_file, "r") as f:
                self.results = json.load(f)
        else:
            self.results = []
    
    def save_results(self):
        with open(self.results_file, "w") as f:
            json.dump(self.results, f, indent=2)

    def load_ratings(self):
        if os.path.exists(self.ratings_file):
            with open(self.ratings_file, "r") as f:
                self.ratings = json.load(f)
        else:
            self.ratings = {path_to_name(cmd): 1200 for cmd in self.engine_paths}

    def save_ratings(self):
        with open(self.ratings_file, "w") as f:
            json.dump(self.ratings, f, indent=2)

    def elo_updater(self, white_name, black_name, winner, k=32):
        elo_white = self.ratings.get(white_name, 1200)
        elo_black = self.ratings.get(black_name, 1200)

        e_white = 1 / (1 + 10 ** ((elo_black - elo_white) / 400)) 
        e_black = 1 - e_white                                     

        if winner == "white":
            s_white, s_black = 1, 0
        elif winner == "black":
            s_white, s_black = 0, 1
        else:
            s_white = s_black = 0.5

        # Update ratings
        self.ratings[white_name] = round(elo_white + k * (s_white - e_white))
        self.ratings[black_name] = round(elo_black + k * (s_black - e_black))

    def compute_elo_from_results(self):
        self.ratings = {path_to_name(cmd): 1200 for cmd in self.engine_paths}
        for match in self.results:
            self.elo_updater(match["white"], match["black"], match["winner"])
        self.save_ratings()

    def get_pair_match_counts(self, engine_names):
        """Count how many matches each pair of engines has played"""
        pair_match_counts = {}
        
        # Initialize counts for all possible pairs
        for i in range(len(engine_names)):
            for j in range(i + 1, len(engine_names)):
                pair = tuple(sorted([engine_names[i], engine_names[j]]))
                pair_match_counts[pair] = 0
                
        for match in self.results:
            white_name = match["white"]
            black_name = match["black"]
            pair = tuple(sorted([white_name, black_name]))
            if pair in pair_match_counts:
                pair_match_counts[pair] += 1
                
        return pair_match_counts

    def select_engines_weighted(self):
        """Select a pair of engines with probability proportional to 1/(1 + matches_played_by_pair)"""
        engine_names = [path_to_name(path) for path in self.engine_paths]
        pair_match_counts = self.get_pair_match_counts(engine_names)
        
        pairs = []
        weights = []
        
        name_to_path = {name: path for name, path in zip(engine_names, self.engine_paths)}
        
        for i in range(len(engine_names)):
            for j in range(i + 1, len(engine_names)):
                name_1, name_2 = engine_names[i], engine_names[j]
                pair_tuple = tuple(sorted([name_1, name_2]))
                matches_played = pair_match_counts.get(pair_tuple, 0)
                pairs.append((name_1, name_2))
                weights.append(1.0 / (1 + matches_played))
                
        weights = np.array(weights)
        if weights.sum() == 0:
            weights = np.ones_like(weights) / len(weights)
        else:
            weights = weights / weights.sum()  # normalize
        
        # Select a pair
        selected_index = np.random.choice(len(pairs), p=weights)
        selected_pair = pairs[selected_index]
        
        a, b = name_to_path[selected_pair[0]], name_to_path[selected_pair[1]]
        if random.random() < 0.5: # randomly assign white/black
            a, b = b, a
        return a, b

    def play_match(self, white_path, black_path, update_elo, verbose, timeout, depth, maxmoves, random_moves, predefined_random_moves=None):
        position = self.starting_position
        engines = [Engine(white_path), Engine(black_path)]

        if verbose:
            print(f"White player: {engines[0].name}")
            print(f"Black player: {engines[1].name}")

        turn = 0  # 0 = white, 1 = black
        colors = ['white', 'black']

        n_moves = 0

        prev_position = self.starting_position
        moves = []
        error = None

        while True:
            try:
                if game_is_ended(position) or n_moves > maxmoves*2:
                    break

                engine = engines[turn]
                color = colors[turn]
                engine.send(f"newgame {position}")
                response = engine.receive(timeout=1)
                if len(response) == 0:
                    raise TimeoutError(f"No newgame response from [{color}].")

                engine.send(f"validmoves")
                validmoves = engine.receive(timeout=1)
                if len(validmoves)==0:
                    raise TimeoutError(f"Engine {engine.name} couldn't find valid moves")
                validmoves = validmoves[0].split(';')

                if n_moves < random_moves:
                    if predefined_random_moves is not None and n_moves < len(predefined_random_moves):
                        move = predefined_random_moves[n_moves]
                    else:
                        move = str(np.random.choice(validmoves))
                else:
                    if depth > 0:
                        engine.send(f"bestmove depth {depth}")
                        move = engine.receive(timeout=MAX_THINK_TIME)
                    else:
                        engine.send(f"bestmove time {seconds_to_hh(timeout)}")
                        move = engine.receive(timeout=timeout + THINK_TOL)
                    if len(move)==0:
                        raise TimeoutError(f"[{color}] timed out or no move")
                    move = move[0]
                    if verbose:
                        print(f"Move {n_moves}: {color} -> {move}")
                    if move not in validmoves:
                        raise TimeoutError(f"Move {move} is not a valid move for {color}")

                moves.append(move)
                engine.send(f"play {move}")
                response = engine.receive(timeout=1)
                if len(response)==0:
                    raise TimeoutError(f"No play from engine {engine.name}.")
                prev_position = position
                position = response[0]
                
                turn ^= 1
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
                print("List of moves:")
                print(moves)
                position = prev_position
                break
                
        if n_moves > maxmoves*2:
            if verbose:
                print("Maximum number of moves exceeded")
            winner = "draw"
        else:
            winner = get_winner_from_gamestate(position)

        if verbose:
            print("Final position: ", position)

        if update_elo:
            if verbose:
                print(f"Winner: {winner}")
            if winner != "other" and n_moves > random_moves:
                self.elo_updater(white_name=engines[0].name, black_name=engines[1].name, winner=winner)

        engines[0].terminate()
        engines[1].terminate()

        result = {
            "white": engines[0].name,
            "black": engines[1].name,
            "winner": winner,
            "final_gamestate": position,
            "datetime": datetime.datetime.now().isoformat(),
            "move_duration": timeout,
            "random_moves": random_moves,
            "initial_moves": moves[:random_moves]
        }
        if error: result["error"] = error
        return result

    def continuous_matches(self, engine_paths_file, update_elo, verbose, timeout, depth, maxmoves, random_moves):
        """Run matches continuously, reloading engine list after each match"""
        match_count = 0
        
        while True:
            # Reload engine paths from file
            try:
                new_engine_paths = []
                with open(engine_paths_file) as f:
                    for path in f:
                        if path.startswith('#'):
                            continue
                        new_engine_paths.append(path.replace('\n','').replace('\\','/'))
                
                # Check if engine list changed
                if new_engine_paths != self.engine_paths:
                    if verbose:
                        print(f"Engine list updated: {len(new_engine_paths)} engines")
                    self.engine_paths = new_engine_paths
                    
                    # Update ratings for new engines
                    for engine_path in self.engine_paths:
                        engine_name = path_to_name(engine_path)
                        if engine_name not in self.ratings:
                            self.ratings[engine_name] = 1200
                
                # Need at least 2 engines
                if len(self.engine_paths) < 2:
                    print("Need at least 2 engines, waiting...")
                    exit(1)
                    
            except FileNotFoundError:
                print(f"Engine paths file {engine_paths_file} not found")
                exit(1)
            except Exception as e:
                print(f"Error reading engine paths: {e}")
                exit(1)
            
            # Select engines weighted by matches played
            try:
                white_path, black_path = self.select_engines_weighted()
                
                match_count += 1
                if verbose:
                    print(f"\n--- Match {match_count} (A) ---")
                
                result1 = self.play_match(white_path=white_path,
                                        black_path=black_path, 
                                        update_elo=update_elo, 
                                        verbose=verbose,
                                        timeout=timeout,
                                        depth=depth,
                                        maxmoves=maxmoves,
                                        random_moves=random_moves)
                
                # Play reverse match
                if verbose:
                    print(f"\n--- Match {match_count} (B) ---")
                
                result2 = self.play_match(white_path=black_path,
                                        black_path=white_path, 
                                        update_elo=update_elo, 
                                        verbose=verbose,
                                        timeout=timeout,
                                        depth=depth,
                                        maxmoves=maxmoves,
                                        random_moves=random_moves,
                                        predefined_random_moves=result1.get("initial_moves"))
                
                self.results.append(result1)
                self.results.append(result2)
                
                self.save_results()
                if update_elo:
                    self.save_ratings()
                    
            except Exception as e:
                print(f"Error during match: {e}")
                time.sleep(1)

    def all_v_all(self, n_matches, update_elo, verbose, timeout, depth, maxmoves, random_moves):
        n_engines = len(self.engine_paths)

        matches_pairs = []
        for _ in range(n_matches):
            for i in range(n_engines):
                for j in range(i + 1, n_engines):
                    matches_pairs.append((i, j))
                    
        if progress_bar and not verbose:
            matches_pairs = tqdm(matches_pairs)

        for i, j in matches_pairs:
            # First match: A vs B
            result1 = self.play_match(white_path=self.engine_paths[i],
                                    black_path=self.engine_paths[j], 
                                    update_elo=update_elo, 
                                    verbose=verbose,
                                    timeout=timeout,
                                    depth=depth,
                                    maxmoves=maxmoves,
                                    random_moves=random_moves)
            
            # Second match: B vs A with same random moves
            result2 = self.play_match(white_path=self.engine_paths[j],
                                    black_path=self.engine_paths[i], 
                                    update_elo=update_elo, 
                                    verbose=verbose,
                                    timeout=timeout,
                                    depth=depth,
                                    maxmoves=maxmoves,
                                    random_moves=random_moves,
                                    predefined_random_moves=result1.get("initial_moves"))
            
            self.results.append(result1)
            self.results.append(result2)
            
            self.save_results()
            self.save_ratings()

def load_engines_with_names(engine_paths_file="logs/paths.txt"):
    engine_paths = []
    names = []
    with open(engine_paths_file) as f:
        for path in f:
            if path.startswith("#"):
                continue
            engine_paths.append(path.replace('\n','').replace('\\','/'))
            names.append(path_to_name(engine_paths[-1]))
    return engine_paths, names

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hive bot arena')
    parser.add_argument("--engine_paths", type=str, default="logs/paths.txt", help="File containing paths to the engine executables")
    parser.add_argument("--n_matches", type=int, default=1, help="Number of matches")
    parser.add_argument("--update-elo", type=bool, default=True, help="If the elo gets updated")
    parser.add_argument("--verbose", action="store_true", help="Display info about matches in real time")
    parser.add_argument("--results-folder", type=str, default="logs/", help="Folder where the matches are stored")
    parser.add_argument("--maxmoves", type=int, default=maxmoves, help="Maximum number of moves per match per engine")
    parser.add_argument("--random-moves", type=int, default=0, help="Number of initial random moves")
    parser.add_argument("--all-v-all", action="store_true", help="Run all-vs-all once instead of continuous selection")
    
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--timeout", type=float, default=5, help="Time per move")
    group.add_argument("--depth", type=int, default=0, help="Depth for engine evaluation (not used in arena)")
    args = parser.parse_args()

    engine_paths, names = load_engines_with_names(args.engine_paths)

    # check wether two different engines have the same path or name
    n_engines = len(engine_paths)
    for i in range(n_engines):
        for j in range(i+1, n_engines):
            name_1, path_1 = names[i], engine_paths[i]
            name_2, path_2 = names[j], engine_paths[j]
            if path_1 == path_2:
                raise ValueError(f"Two different engines at the same location: \"{path_1}\"")
            elif name_1 == name_2:
                raise ValueError(f"Names of different engines are the same: \n Name of engine at location \"{path_1}\" is {name_1} \n Name of engine at location \"{path_2}\" is {name_2}")


    arena = HiveArena(engine_paths, results_folder=args.results_folder)
    
    if not args.all_v_all:
        arena.continuous_matches(engine_paths_file=args.engine_paths,
                                update_elo=args.update_elo, 
                                verbose=args.verbose, 
                                timeout=args.timeout, 
                                depth=args.depth,
                                maxmoves=args.maxmoves,
                                random_moves=args.random_moves)
    else:
        arena.all_v_all(n_matches=args.n_matches, 
                        update_elo=args.update_elo, 
                        verbose=args.verbose, 
                        timeout = args.timeout, 
                        depth=args.depth,
                        maxmoves = args.maxmoves,
                        random_moves=args.random_moves)

"""
    TODO:
    - Assumes that all made moves are valid
    - Probably the elo could be simply recomputed from scratch every time
    - If a bot crashes, the arena might end up in an infinite loop of trying to restart it and keep getting the same error
"""
