import functools
from io import text_encoding
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
import re
progress_bar = True
try:
    from tqdm import tqdm
except:
    progress_bar = False
from multiprocessing import Pool, cpu_count

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

    def command(self, text, timout=None):
        self.send(text)
        return self.receive(timeout=timout)

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
        self.stderr_log_file.close()
        self.proc.terminate()
        try:
            self.proc.wait(timeout=1)
        except subprocess.TimeoutExpired:
            self.proc.kill()

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

def get_n_moves_from_position(position: str):
    moves_str = position.split(';')[2]
    return int(re.search(r"\[(\d+)\]", moves_str).group(1))

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

    def get_match_counts(self):
        """Count how many matches each engine has played"""
        match_counts = {}
        for engine_path in self.engine_paths:
            engine_name = path_to_name(engine_path)
            match_counts[engine_name] = 0
        
        for match in self.results:
            white_name = match["white"]
            black_name = match["black"]
            if white_name in match_counts:
                match_counts[white_name] += 1
            if black_name in match_counts:
                match_counts[black_name] += 1
        
        return match_counts

    def select_engines_weighted(self):
        """Select two engines with probability proportional to 1/(1 + matches_played)"""
        match_counts = self.get_match_counts()
        
        # Calculate weights for each engine
        weights = []
        for engine_path in self.engine_paths:
            engine_name = path_to_name(engine_path)
            matches_played = match_counts.get(engine_name, 0)
            weight = 1.0 / (1 + matches_played)
            weights.append(weight)
        
        weights = np.array(weights)
        weights = weights / weights.sum()  # normalize
        
        # Select two different engines
        selected_indices = np.random.choice(len(self.engine_paths), size=2, replace=False, p=weights)
        a, b = self.engine_paths[selected_indices[0]], self.engine_paths[selected_indices[1]]
        if random.random() < 0.5: # avoid a new engine with high weight always being white/black
            a, b = b, a
        return a, b

    def play_match(self, white_path, black_path, update_elo, verbose, timeout, depth, maxmoves, random_moves, from_position=None):
        position = self.starting_position
        if from_position:
            position = from_position
        engines = [Engine(white_path), Engine(black_path)]

        if verbose:
            print(f"White player: {engines[0].name}")
            print(f"Black player: {engines[1].name}")

        n_moves = get_n_moves_from_position(position)

        turn = n_moves & 1  # 0 = white, 1 = black
        colors = ['white', 'black']

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
            "random_moves": random_moves
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
                    print(f"\n--- Match {match_count} ---")
                
                result = self.play_match(white_path=white_path,
                                        black_path=black_path, 
                                        update_elo=update_elo, 
                                        verbose=verbose,
                                        timeout=timeout,
                                        depth=depth,
                                        maxmoves=maxmoves,
                                        random_moves=random_moves)
                self.results.append(result)
                
                self.save_results()
                if update_elo:
                    self.save_ratings()
                    
            except Exception as e:
                print(f"Error during match: {e}")
                time.sleep(1)

    def all_v_all(self, n_matches, update_elo, verbose, timeout, depth, maxmoves, random_moves):
        n_engines = len(self.engine_paths)

        matches = []
        for _ in range(n_matches):
            for i in range(n_engines):
                for j in range(n_engines):
                    if i != j:
                        matches.append((i, j))
        if progress_bar and not verbose:
            matches = tqdm(matches)

        for i, j in matches:
            result = self.play_match(white_path=self.engine_paths[i],
                                    black_path=self.engine_paths[j], 
                                    update_elo=update_elo, 
                                    verbose=verbose,
                                    timeout=timeout,
                                    depth=depth,
                                    maxmoves=maxmoves,
                                    random_moves=random_moves)
            self.results.append(result)
            
            self.save_results()
            self.save_ratings()

class Player:
    def __init__(self, name, path):
        self.name = name
        self.path = path
        self.score = 0.0
        self.opponents = set()

def update_victories(players, victories, rev, position, swiss):
            
            result = swiss.arena.play_match(players[0].path, players[1].path, update_elo=False, verbose=swiss.verbose,
                                           timeout=swiss.timeout, depth=swiss.depth, maxmoves=200, random_moves=4, from_position=position)
            if result["winner"] == "white":
                victories[0^rev] += 1
            elif result["winner"] == "black":
                victories[1^rev] += 1
            elif result["winner"] == "draw" or result["winner"] == "other":
                victories[0] += 0.5
                victories[1] += 0.5
            else:
                raise ValueError(f"Error: unrecognised value for result[\"winner\"], got value {result['winner']}")
            
            return result

def play_black_white(position, player1, player2, swiss):
    victories = [0,0]

    results = [update_victories(players=[player1,player2], victories=victories, rev=0, position=position, swiss=swiss),
               update_victories(players=[player2,player1], victories=victories, rev=1, position=position, swiss=swiss)]

    return [victories[0], victories[1], results]

def play_match(data, swiss):
    player1, player2, position = data[0], data[1]

class SwissTournament:

    def load_results(self):
        if os.path.exists(self.results_file):
            with open(self.results_file, "r") as f:
                self.match_results = json.load(f)
        else:
            self.match_results = []

    # Ok probabilmente era meglio fare una funzione in HiveArena ma sticazzi dai
    def __init__(self, engine_paths, fair_positions_path=None, fair_positions_number=None, timeout=5, depth=0, verbose=False, processses=1, results_folder="logs/"):
        if len(engine_paths) & (len(engine_paths) - 1) != 0:
            raise ValueError("Number of engines must be a power of 2")

        self.timeout = timeout
        self.depth = depth
        self.verbose = verbose
        self.processes = processses
        self.results_folder = results_folder
        self.results_file = os.path.join(results_folder, "results.json")
        self.load_results()

        engine_names = []

        for path in engine_paths:
            engine = Engine(path)
            engine_names.append((engine.name, path))
            engine.terminate()
        
        self.players = {}
        for name, path in engine_names:
            self.players[name] = Player(name=name, path=path)
        
        self.rounds = []
        self.current_round = 0
        self.arena = HiveArena(engine_paths)

        self.starting_positions = [self.arena.starting_position]
        if fair_positions_path:
            if not fair_positions_number:
                raise ValueError("Error provide the number of fair positions to be selected from the fair positions file")
            self.starting_positions = []
            with open("logs/fair_positions.json") as f:
                scores = json.load(f)

            scores_list = sorted([(abs(score["evaluation"]), score["position"]) for score in scores])

            for i in range(min(len(scores_list), fair_positions_number)):
                self.starting_positions.append(scores_list[i][1])


    def get_optimal_rounds(self):
        n = len(self.players)
        if n <= 8:
            return min(6, n - 1)
        elif n <= 16:
            return min(7, n - 1)
        else:
            return min(8, n - 1)
    
    def create_pairings(self):
        players_list = list(self.players.values())
        
        players_list.sort(key=lambda p: (-p.score, p.name))

        best_score= players_list[0].score
        
        pairings = []
        unpaired = players_list[:]
        
        while len(unpaired) >= 2:
            player1 = unpaired.pop(0)

            if player1.score <= best_score - 3:
                break
            
            best_opponent = None
            best_index = -1
            
            for i, player2 in enumerate(unpaired):
                if player2.name not in player1.opponents:
                    best_opponent = player2
                    best_index = i
                    break
            
            if best_opponent is None:
                best_opponent = unpaired[0]
                best_index = 0
            
            unpaired.pop(best_index)
            pairings.append((player1, best_opponent))
            
            player1.opponents.add(best_opponent.name)
            best_opponent.opponents.add(player1.name)
        
        if unpaired:
            pairings.append((unpaired[0], "BYE"))
            self.players[unpaired[0].name].score += 0.5
        
        return pairings
    
    def create_matches_safe(self, p1, p2):
        
        to_play = []
        for position in self.starting_positions:
            to_play.append([p1, p2, position])
            to_play.append([p2, p1, position])
        
        return to_play

    def create_matches_to_play(self, pairings):
        
        to_play = []
        for p1, p2 in pairings:
            if p2 == "BYE":
                continue

            to_play += self.create_matches_safe(p1, p2)

        return to_play
    
    def schedule_round(self):
        self.current_round += 1
        pairings = self.create_pairings()
        self.rounds.append(pairings)
        return pairings
    
    def record_results(self, results):
        for player1, player2, result in results:
            if player2 == "BYE":
                continue 
                
            self.players[player1.name].score += result
            self.players[player2.name].score += (1.0 - result)
    
    def get_standings(self):
        standings = []
        for p in self.players.values():
            standings.append((p.name, p.score))
        standings.sort(key=lambda x: (-x[1], x[0]))
        return standings
    
    def print_standings(self):
        standings = self.get_standings()
        print(f"\n=== STANDINGS AFTER ROUND {self.current_round} ===")
        for i, standing in enumerate(standings, 1):
            name, score = standing
            print(f"{i:2d}. {name:<15} {score:4.1f} points")
    
    def play_round_safe(self, player1, player2):

        victories1 = 0
        victories2 = 0
        match_results = []

        worker_func = functools.partial(
            play_black_white,
            player1=player1,
            player2=player2,
            swiss=self
        )
            
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['OPENBLAS_NUM_THREADS'] = '1'
        os.environ['MKL_NUM_THREADS'] = '1'
        os.environ['NUMEXPR_NUM_THREADS'] = '1'
        
        starting_positions = self.starting_positions
        with Pool(processes=self.processes) as pool:
            
            results = list(pool.imap(worker_func, starting_positions))
            
            for result in results:
                victories1 += result[0]
                victories2 += result[1]
                match_results += result[2]


        return (victories1, victories2, match_results)

    def run_tournament_simulation(self):
        matches_results = self.match_results
        optimal_rounds = self.get_optimal_rounds()
        print(f"Starting Swiss tournament with {len(self.players)} engines")
        print(f"Recommended rounds: {optimal_rounds}")

        standings_history = []

        for _round_num in range(1, optimal_rounds + 1):
            pairings = self.schedule_round()
            
            

            results = []
            for pairing in pairings:
                p1, p2 = pairing
                if p2 != "BYE":
                    
                    v1, v2, match_results = self.play_round_safe(p1, p2)
                    matches_results += match_results
                    if v1 > v2:
                        result = 1.0  # p1 wins
                    elif v1 == v2:
                        result = 0.5  # draw
                    else:
                        result = 0.0  # p2 wins
                    results.append((p1, p2, result))
                with open(os.path.join(self.results_folder, "results.json"), "w") as f:
                    json.dump(matches_results, f, indent=2) # these are the results of each match played in the swiss torunament
        
            
            # Record results
            self.record_results(results)
            self.print_standings()
            standings_history.append(self.get_standings())
            with open(os.path.join(self.results_folder, "swiss_results.json"), "w") as f:
                json.dump(standings_history, f)
            
        if self.verbose:
            time.sleep(1)

        print("\n=== FINAL RANKING ===")
        self.print_standings()
        return (self.get_standings(), matches_results)

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
    group_tournament_type = parser.add_mutually_exclusive_group()
    group_tournament_type.add_argument("--continuous", action="store_true", help="Run continuously, reloading engine list after each match")
    group_tournament_type.add_argument("--swiss", action="store_true", help="Run Swiss tournament ")
    parser.add_argument("--fair-positions-path", type=str, default="logs/fair_positions.json", help="Path to json of evaluated positions to pick for first position")
    parser.add_argument("--fair-positions-number", type=int, default=5, help="Number of positions to pick")
    
    group_eval_type = parser.add_mutually_exclusive_group(required=False)
    group_eval_type.add_argument("--timeout", type=float, default=5, help="Time per move")
    group_eval_type.add_argument("--depth", type=int, default=0, help="Depth for engine evaluation (not used in arena)")

    parser.add_argument("--processes", type=int, default=1, help="Number of parallel processes for the arena (only works for swiss)")
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

    if args.swiss:
        tournament = SwissTournament(engine_paths=engine_paths, fair_positions_path=args.fair_positions_path, fair_positions_number=args.fair_positions_number, timeout=args.timeout, depth=args.depth, verbose=args.verbose, processses=args.processes, results_folder=args.results_folder)
        ratings = tournament.run_tournament_simulation()
        with open(os.path.join(args.results_folder, "swiss_results.json"), "w") as f:
            json.dump(ratings[0], f)
        with open(os.path.join(args.results_folder, "results.json"), "w") as f:
            json.dump(ratings[1], f, indent=2) # these are the results of each match played in the swiss torunament
        exit(0)


    arena = HiveArena(engine_paths, results_folder=args.results_folder)
    
    if args.continuous:
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
