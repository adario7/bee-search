import subprocess
import threading
import queue
import time
import datetime
import json
import os
import argparse
progress_bar = True
try:
    from tqdm import tqdm
except:
    progress_bar = False
timeout = 5  # default timeout per move
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

    def play_match(self, white_path, black_path, update_elo = False, verbose = False, timeout=timeout, maxmoves=maxmoves):
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

        while True:
            if game_is_ended(position) or n_moves > maxmoves*2:
                break
            
            try:
                engine = engines[turn]
                color = colors[turn]
                engine.send(f"newgame {position}")
                response = engine.receive(timeout=1)
                if len(response) == 0:
                    raise TimeoutError(f"No newgame response from [{color}].")

                engine.send(f"bestmove time {seconds_to_hh(timeout)}")
                move = engine.receive(timeout=timeout + 1)

                if len(move)==0:
                    raise TimeoutError(f"[{color}] timed out or no move")
                    
                if verbose:
                    print(f"Move {n_moves}: {color} -> {move[0]}")
                
                moves.append(move[0])
                engine.send(f"play {move[0]}")
                response = engine.receive(timeout=1)
                if len(response)==0:
                    raise TimeoutError(f"No play from [{color}].")
                
                prev_position = position
                position = response[0]
                
                turn ^= 1
                n_moves += 1
            except TimeoutError:
                print(f"Unable to connnect with engine {engine.name}")
                result = input("If you want to go on with the next match type \"Y\": ")
                if result.lower() == 'y':
                    break
                else:
                    exit(0)
            except IndexError:
                print("Probably an invalid move was made in this position:")
                print(prev_position)
                print("List of moves:")
                print(moves)
                position = prev_position
                break
                
        if n_moves > maxmoves*2:
            if verbose:
                print("Maximum number of moves exceeded")
        if verbose:
            print("Final position: ", position)

        if update_elo:
            winner = get_winner_from_gamestate(position)
            if winner != "other":
                self.elo_updater(white_name=engines[0].name, black_name=engines[1].name, winner=winner)

        engines[0].terminate()
        engines[1].terminate()

        return {
            "white": engines[0].name,
            "black": engines[1].name,
            "winner": winner,
            "final_gamestate": position,
            "datetime": datetime.datetime.now().isoformat(),
            "move_duration": timeout
        }

    def elo_updater(self, white_name, black_name, winner, k=32):
        elo_white = self.ratings.get(white_name, 1200)
        elo_black = self.ratings.get(black_name, 1200)

        e_white = 1 / (1 + 10 ** ((elo_white - elo_black) / 400)) 
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

    def all_v_all(self, n_matches = 1, update_elo=False, verbose=False, timeout=timeout, maxmoves=maxmoves):
        n_engines = len(self.engine_paths)

        if progress_bar and not verbose:
            matches = []
            for _ in range(n_matches):
                for i in range(n_engines):
                    for j in range(n_engines):
                        if i != j:
                            matches.append((i, j))
            for i, j in tqdm(matches):
                result = self.play_match(white_path=self.engine_paths[i],
                                        black_path=self.engine_paths[j], 
                                        update_elo=update_elo, 
                                        verbose=verbose,
                                        timeout=timeout,
                                        maxmoves=maxmoves)
                self.results.append(result)
                
                self.save_results()
                self.save_ratings()
        else:
            for _ in range(n_matches):
                for i in range(n_engines):
                    for j in range(n_engines):
                        if i != j:
                            result = self.play_match(white_path=self.engine_paths[i],
                                                    black_path=self.engine_paths[j], 
                                                    update_elo=update_elo, 
                                                    verbose=verbose,
                                                    timeout=timeout,
                                                    maxmoves=maxmoves)
                            self.results.append(result)

                            self.save_results()
                            self.save_ratings()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hive bot arena')
    parser.add_argument("--engine_paths", type=str, default="logs/paths.txt", help="File containing paths to the engine executables")
    parser.add_argument("--n_matches", type=int, default=1, help="Number of matches")
    parser.add_argument("--update_elo", type=bool, default=True, help="If the elo gets updated")
    parser.add_argument("--verbose", type=bool, default=False, help="Display info about matches in real time")
    parser.add_argument("--results_folder", type=str, default="logs/", help="Folder where the matches are stored")
    parser.add_argument("--timeout", type=float, default=timeout, help="Time per move")
    parser.add_argument("--maxmoves", type=int, default=maxmoves, help="Maximum number of moves per match per engine")
    args = parser.parse_args()

    engine_paths = []
    names = []
    with open(args.engine_paths) as f:
        for path in f:
            engine_paths.append(path.replace('\n','').replace('\\','/')) # sì ok, uso ancora Windows 
            names.append(path_to_name(engine_paths[-1]))

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
    arena.all_v_all(n_matches=args.n_matches, 
                    update_elo=args.update_elo, 
                    verbose=args.verbose, 
                    timeout = args.timeout, 
                    maxmoves = args.maxmoves)

"""
    TODO:
    - Assumes that all made moves are valid
    - Probably the elo could be simply recomputed from scratch every time
    - If a bot crashes, the arena might end up in an infinite loop of trying to restart it and keep getting the same error
"""