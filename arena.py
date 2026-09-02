import subprocess
import threading
import queue
import time
import datetime
import json
import os
import argparse
import random
import re
import shutil
import sys
import fcntl
import numpy as np

MAX_THINK_TIME = 120  # maximum time for the engine to think (used if depth is used instead of timeout)
THINK_TOL = 0.1  # tolerance for timeout
maxmoves = 200  # maximum number of moves per player per game

_name_cache = {}


class Engine:
    def __init__(self, path, name=None, log_name=None, log_folder="logs", configure_options=True):
        self.path = path
        self.log_folder = log_folder
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

        info = self.receive(timeout=2)
        if info and len(info) > 0 and 'id ' in info[0]:
            self.name = info[0].split('id ')[1]
        elif info and len(info) > 0:
            self.name = info[0]
        else:
            self.name = os.path.basename(path)
        self.id_name = self.name

        safe_name = re.sub(r'[^\w\-_\.]', '_', self.name)
        if log_name:
            self.log_name = log_name
        else:
            self.log_name = f"{safe_name}_{os.getpid()}_{threading.get_ident()}_{random.randint(1000, 9999)}"

        self.stderr_log_path = os.path.join(self.log_folder, f"{self.log_name}_stderr.log")
        self.stderr_log_file = None
        self._stderr_queue = queue.Queue()
        threading.Thread(target=self._read_stderr, daemon=True).start()

        if configure_options:
            # nokamute format
            self.command("options set NumThreads 1", timeout=0.05)
            self.command("options set TableSizeMiB 256", timeout=0.05)
            # mzinga format
            self.command("options set MaxHelperThreads None", timeout=0.05)
            self.command("options set TranspositionTableSizeMB 1024", timeout=0.05)
            time.sleep(0.05)
            self.drain_stdout()

    def _reader(self):
        try:
            for line in self.proc.stdout:
                self._stdout_queue.put(line)
        except Exception:
            pass

    def _read_stderr(self):
        try:
            for line in self.proc.stderr:
                self._stderr_queue.put(line)
                if self.stderr_log_file is None:
                    os.makedirs(os.path.dirname(self.stderr_log_path), exist_ok=True)
                    self.stderr_log_file = open(self.stderr_log_path, "w")
                self.stderr_log_file.write(line)
                self.stderr_log_file.flush()
        except Exception:
            pass

    def drain_stdout(self):
        while not self._stdout_queue.empty():
            try:
                self._stdout_queue.get_nowait()
            except queue.Empty:
                break

    def send(self, text):
        if self.proc.stdin:
            try:
                self.proc.stdin.write(text + "\n")
                self.proc.stdin.flush()
            except Exception:
                pass

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
        try:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=1)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        except Exception:
            pass
        if self.stderr_log_file:
            try:
                self.stderr_log_file.close()
            except Exception:
                pass
            self.stderr_log_file = None

    def restart(self):
        print(f"Try restarting the engine [{self.name}]", flush=True)
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
    time_val = int(seconds)
    s = time_val % 60
    time_val //= 60
    m = time_val % 60
    h = time_val // 60
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
        raise IndexError("Status has length 1")

    if status != 'NotStarted' and status != 'InProgress':
        return True
    return False


def path_to_name(path):
    if path in _name_cache:
        return _name_cache[path]
    engine = Engine(path, configure_options=False)
    name = engine.name
    engine.terminate()
    _name_cache[path] = name
    return name


class ArenaReporter:
    """Thread-safe reporter for game progress and completions."""
    def __init__(self, verbose=True, jobs=1):
        self.verbose = verbose
        self.jobs = jobs
        self.active_games = {}
        self.completed_count = 0
        self.lock = threading.RLock()
        self.is_tty = sys.stdout.isatty()
        self.rendered_lines = 0

    def _clear_live(self):
        if not self.is_tty or self.rendered_lines == 0:
            return
        sys.stdout.write(f"\033[{self.rendered_lines}A")
        for _ in range(self.rendered_lines):
            sys.stdout.write("\033[2K\r\033[B")
        sys.stdout.write(f"\033[{self.rendered_lines}A")
        sys.stdout.flush()
        self.rendered_lines = 0

    def _render_live(self):
        if not self.is_tty or not self.verbose:
            return
        cols = shutil.get_terminal_size().columns
        lines = []
        if self.jobs > 1:
            lines.append(f"--- In Progress ({len(self.active_games)}/{self.jobs} games) ---")
        for gid, g in sorted(self.active_games.items(), key=lambda x: (x[1].get('worker_id', 0), x[0])):
            dur = int(time.time() - g['start'])
            last = f" | Last: {g['last_move']}" if g['last_move'] else ""
            line = f"  [{gid}] {g['white']} (W) vs {g['black']} (B) | Moves: {g['moves']:2d} ({g['color']}){last} ({dur}s)"
            lines.append(line[:cols - 1])

        for line in lines:
            sys.stdout.write("\033[2K\r" + line + "\n")
        sys.stdout.flush()
        self.rendered_lines = len(lines)

    def game_started(self, gid, white, black, worker_id=None):
        with self.lock:
            self.active_games[gid] = {
                'white': white,
                'black': black,
                'moves': 0,
                'color': 'white',
                'last_move': None,
                'start': time.time(),
                'worker_id': worker_id
            }
            if not self.verbose:
                return
            if self.is_tty:
                self._clear_live()
                self._render_live()
            else:
                print(f"[{gid}] Started: {white} (W) vs {black} (B)", flush=True)

    def game_move(self, gid, moves, color, move):
        with self.lock:
            if gid in self.active_games:
                self.active_games[gid]['moves'] = moves
                self.active_games[gid]['color'] = color
                self.active_games[gid]['last_move'] = f"{color} -> {move}"
            if not self.verbose:
                return
            if self.is_tty:
                self._clear_live()
                self._render_live()
            elif self.jobs == 1 or moves % 5 == 0:
                print(f"[{gid}] Move {moves}: {color} -> {move}", flush=True)

    def game_completed(self, gid, white, black, winner, moves, elo_info=None, error=None):
        with self.lock:
            if gid in self.active_games:
                del self.active_games[gid]
            self.completed_count += 1
            if not self.verbose:
                return
            if self.is_tty:
                self._clear_live()

            elo_str = ""
            if elo_info:
                w_name, w_elo, w_delta = elo_info["white"]
                b_name, b_elo, b_delta = elo_info["black"]
                elo_str = f" | Elo: {w_name} {w_elo} ({w_delta:+d}), {b_name} {b_elo} ({b_delta:+d})"
            err_str = f" | Error: {error}" if error else ""

            print(f"[{gid} COMPLETED] {white} (W) vs {black} (B) -> Winner: {winner} in {moves} moves{elo_str}{err_str}", flush=True)

            if self.is_tty and self.active_games:
                self._render_live()

    def log_message(self, msg):
        with self.lock:
            if self.is_tty and self.rendered_lines > 0:
                self._clear_live()
            print(msg, flush=True)
            if self.is_tty and self.active_games:
                self._render_live()

    def finish(self):
        with self.lock:
            if self.is_tty and self.rendered_lines > 0:
                self._clear_live()
            print(f"\nArena finished. Total completed games: {self.completed_count}", flush=True)


class HiveArena:
    def __init__(self, engine_paths, results_folder="logs/", jobs=1):
        self.engine_paths = engine_paths
        self.results_folder = results_folder
        os.makedirs(self.results_folder, exist_ok=True)
        self.results_file = os.path.join(results_folder, "results.json")
        self.ratings_file = os.path.join(results_folder, "ratings.json")
        self.lock_file = os.path.join(results_folder, ".arena.lock")
        self.starting_position = "Base+MLP;NotStarted;White[1]"
        self.jobs = jobs
        self.lock = threading.RLock()
        self.match_counter = 0

        self.load_results()
        self.load_ratings()

    def _atomic_save_json(self, filepath, data):
        """Safely write JSON to a temp file and atomically replace target file, with flock."""
        tmp_file = f"{filepath}.tmp.{os.getpid()}_{threading.get_ident()}_{random.randint(1000, 9999)}"
        try:
            with open(tmp_file, "w") as f:
                json.dump(data, f, indent=2)
                f.flush()
                os.fsync(f.fileno())

            lock_fd = None
            try:
                lock_fd = open(self.lock_file, "w")
                fcntl.flock(lock_fd.fileno(), fcntl.LOCK_EX)
            except Exception:
                pass

            try:
                os.replace(tmp_file, filepath)
            finally:
                if lock_fd:
                    try:
                        fcntl.flock(lock_fd.fileno(), fcntl.LOCK_UN)
                        lock_fd.close()
                    except Exception:
                        pass
        except Exception as e:
            if os.path.exists(tmp_file):
                try:
                    os.remove(tmp_file)
                except Exception:
                    pass
            raise e

    def load_results(self):
        with self.lock:
            if os.path.exists(self.results_file):
                with open(self.results_file, "r") as f:
                    self.results = json.load(f)
            else:
                self.results = []

    def save_results(self):
        self._atomic_save_json(self.results_file, self.results)

    def load_ratings(self):
        with self.lock:
            if os.path.exists(self.ratings_file):
                with open(self.ratings_file, "r") as f:
                    self.ratings = json.load(f)
            else:
                self.ratings = {path_to_name(cmd): 1200 for cmd in self.engine_paths}

    def save_ratings(self):
        self._atomic_save_json(self.ratings_file, self.ratings)

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

        new_white = round(elo_white + k * (s_white - e_white))
        new_black = round(elo_black + k * (s_black - e_black))

        self.ratings[white_name] = new_white
        self.ratings[black_name] = new_black

        return {
            "white": (white_name, new_white, new_white - elo_white),
            "black": (black_name, new_black, new_black - elo_black),
        }

    def record_game_result(self, result, update_elo=True):
        with self.lock:
            self.results.append(result)
            elo_info = None
            winner = result.get("winner")
            n_moves = result.get("n_moves", 0)
            random_moves = result.get("random_moves", 0)
            if update_elo and winner != "other" and n_moves > random_moves:
                elo_info = self.elo_updater(
                    white_name=result["white"],
                    black_name=result["black"],
                    winner=winner
                )
            self.save_results()
            if update_elo:
                self.save_ratings()
            return elo_info

    def compute_elo_from_results(self):
        with self.lock:
            self.ratings = {path_to_name(cmd): 1200 for cmd in self.engine_paths}
            for match in self.results:
                self.elo_updater(match["white"], match["black"], match["winner"])
            self.save_ratings()

    def get_pair_match_counts(self, engine_names):
        pair_match_counts = {}
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
        with self.lock:
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
                weights = weights / weights.sum()

            selected_index = np.random.choice(len(pairs), p=weights)
            selected_pair = pairs[selected_index]

            a, b = name_to_path[selected_pair[0]], name_to_path[selected_pair[1]]
            if random.random() < 0.5:
                a, b = b, a
            return a, b

    def play_match(self, white_path, black_path, update_elo=True, verbose=True, timeout=5, depth=0,
                   maxmoves=maxmoves, random_moves=0, predefined_random_moves=None, game_id=None,
                   reporter=None, worker_id=None):
        game_id_str = str(game_id) if game_id is not None else f"{random.randint(1000, 9999)}"
        position = self.starting_position

        white_clean = re.sub(r'[^\w\-_\.]', '_', os.path.basename(white_path))
        black_clean = re.sub(r'[^\w\-_\.]', '_', os.path.basename(black_path))

        engines = [
            Engine(white_path, log_name=f"game_{game_id_str}_{white_clean}_white", log_folder=self.results_folder),
            Engine(black_path, log_name=f"game_{game_id_str}_{black_clean}_black", log_folder=self.results_folder)
        ]

        try:
            if reporter:
                reporter.game_started(game_id_str, engines[0].name, engines[1].name, worker_id=worker_id)
            elif verbose:
                print(f"White player: {engines[0].name}", flush=True)
                print(f"Black player: {engines[1].name}", flush=True)

            turn = 0  # 0 = white, 1 = black
            colors = ['white', 'black']
            n_moves = 0
            prev_position = self.starting_position
            moves = []
            error = None

            while True:
                try:
                    if game_is_ended(position) or n_moves > maxmoves * 2:
                        break

                    engine = engines[turn]
                    color = colors[turn]
                    engine.drain_stdout()
                    engine.send(f"newgame {position}")
                    response = engine.receive(timeout=2)
                    if len(response) == 0:
                        raise TimeoutError(f"No newgame response from [{color}].")

                    engine.drain_stdout()
                    engine.send("validmoves")
                    validmoves = engine.receive(timeout=2)
                    if len(validmoves) == 0:
                        raise TimeoutError(f"Engine {engine.name} couldn't find valid moves")
                    validmoves = validmoves[0].split(';')

                    if n_moves < random_moves:
                        if predefined_random_moves is not None and n_moves < len(predefined_random_moves):
                            move = predefined_random_moves[n_moves]
                        else:
                            move = str(np.random.choice(validmoves))
                    else:
                        engine.drain_stdout()
                        if depth > 0:
                            engine.send(f"bestmove depth {depth}")
                            move = engine.receive(timeout=MAX_THINK_TIME)
                        else:
                            engine.send(f"bestmove time {seconds_to_hh(timeout)}")
                            move = engine.receive(timeout=timeout + THINK_TOL)
                        if len(move) == 0:
                            raise TimeoutError(f"[{color}] timed out or no move")
                        move = move[0]
                        if reporter:
                            reporter.game_move(game_id_str, n_moves + 1, color, move)
                        elif verbose:
                            print(f"Move {n_moves}: {color} -> {move}", flush=True)
                        if move not in validmoves:
                            raise TimeoutError(f"Move {move} is not a valid move for {color}")

                    moves.append(move)
                    engine.drain_stdout()
                    engine.send(f"play {move}")
                    response = engine.receive(timeout=2)
                    if len(response) == 0:
                        raise TimeoutError(f"No play from engine {engine.name}.")
                    prev_position = position
                    position = response[0]

                    turn ^= 1
                    n_moves += 1
                except TimeoutError as e:
                    error = str(e)
                    if reporter:
                        reporter.log_message(f"[{game_id_str}] TimeoutError: {e}")
                    else:
                        print(e, flush=True)
                        print(f"Unable to connect with engine {engine.name}", flush=True)
                    break
                except IndexError as e:
                    error = str(e)
                    if reporter:
                        reporter.log_message(f"[{game_id_str}] IndexError: {e}")
                    else:
                        print(e, flush=True)
                        print("Probably an invalid move was made in this position:", flush=True)
                        print(prev_position, flush=True)
                        print("List of moves:", flush=True)
                        print(moves, flush=True)
                    position = prev_position
                    break
                except Exception as e:
                    error = str(e)
                    if reporter:
                        reporter.log_message(f"[{game_id_str}] Error: {e}")
                    else:
                        print(f"Error during match: {e}", flush=True)
                    break

            if n_moves > maxmoves * 2:
                if verbose and not reporter:
                    print("Maximum number of moves exceeded", flush=True)
                winner = "draw"
            else:
                winner = get_winner_from_gamestate(position)

            if verbose and not reporter:
                print("Final position: ", position, flush=True)

            white_name = engines[0].name
            black_name = engines[1].name

            result = {
                "white": white_name,
                "black": black_name,
                "winner": winner,
                "final_gamestate": position,
                "datetime": datetime.datetime.now().isoformat(),
                "move_duration": timeout,
                "random_moves": random_moves,
                "initial_moves": moves[:random_moves],
                "n_moves": n_moves
            }
            if error:
                result["error"] = error
            return result
        finally:
            engines[0].terminate()
            engines[1].terminate()

    def continuous_matches(self, engine_paths_file, update_elo=True, verbose=True, timeout=5, depth=0,
                           maxmoves=maxmoves, random_moves=0, jobs=1):
        """Run matches continuously using `jobs` parallel worker threads."""
        reporter = ArenaReporter(verbose=verbose, jobs=jobs)
        stop_event = threading.Event()
        last_mtime = 0

        reporter.log_message("=== Hive Bot Arena ===")
        reporter.log_message(f"Workers: {jobs} parallel games")
        reporter.log_message(f"Timeout: {timeout}s | Max Moves: {maxmoves} | Random Opening: {random_moves}")
        reporter.log_message(f"Results folder: {self.results_folder}")
        reporter.log_message(f"Engines ({len(self.engine_paths)}): {', '.join([path_to_name(p) for p in self.engine_paths])}\n")

        def reload_engines_if_needed():
            nonlocal last_mtime
            with self.lock:
                try:
                    if not os.path.exists(engine_paths_file):
                        reporter.log_message(f"Engine paths file {engine_paths_file} not found")
                        return False

                    mtime = os.path.getmtime(engine_paths_file)
                    if mtime == last_mtime and len(self.engine_paths) >= 2:
                        return True
                    last_mtime = mtime

                    new_engine_paths = []
                    with open(engine_paths_file) as f:
                        for path in f:
                            path = path.strip()
                            if not path or path.startswith('#'):
                                continue
                            new_engine_paths.append(path.replace('\\', '/'))

                    if new_engine_paths != self.engine_paths:
                        reporter.log_message(f"Engine list updated: {len(new_engine_paths)} engines")
                        self.engine_paths = new_engine_paths

                        for engine_path in self.engine_paths:
                            engine_name = path_to_name(engine_path)
                            if engine_name not in self.ratings:
                                self.ratings[engine_name] = 1200

                    if len(self.engine_paths) < 2:
                        reporter.log_message("Need at least 2 engines!")
                        return False
                    return True
                except Exception as e:
                    reporter.log_message(f"Error reading engine paths: {e}")
                    return False

        def worker_loop(worker_id):
            while not stop_event.is_set():
                if not reload_engines_if_needed():
                    time.sleep(1)
                    continue

                with self.lock:
                    self.match_counter += 1
                    match_idx = self.match_counter

                try:
                    white_path, black_path = self.select_engines_weighted()
                except Exception as e:
                    reporter.log_message(f"Error selecting engines: {e}")
                    time.sleep(1)
                    continue

                game_a_id = f"{match_idx}A"
                game_b_id = f"{match_idx}B"

                # Match A: white vs black
                try:
                    res1 = self.play_match(
                        white_path=white_path,
                        black_path=black_path,
                        update_elo=update_elo,
                        verbose=verbose,
                        timeout=timeout,
                        depth=depth,
                        maxmoves=maxmoves,
                        random_moves=random_moves,
                        game_id=game_a_id,
                        reporter=reporter,
                        worker_id=worker_id
                    )
                    elo_info1 = self.record_game_result(res1, update_elo=update_elo)
                    reporter.game_completed(
                        game_a_id, res1["white"], res1["black"], res1["winner"],
                        res1.get("n_moves", 0), elo_info=elo_info1, error=res1.get("error")
                    )
                except Exception as e:
                    reporter.log_message(f"Error during match {game_a_id}: {e}")
                    time.sleep(1)
                    continue

                if stop_event.is_set():
                    break

                # Match B: reverse colors with same initial random moves
                try:
                    res2 = self.play_match(
                        white_path=black_path,
                        black_path=white_path,
                        update_elo=update_elo,
                        verbose=verbose,
                        timeout=timeout,
                        depth=depth,
                        maxmoves=maxmoves,
                        random_moves=random_moves,
                        predefined_random_moves=res1.get("initial_moves"),
                        game_id=game_b_id,
                        reporter=reporter,
                        worker_id=worker_id
                    )
                    elo_info2 = self.record_game_result(res2, update_elo=update_elo)
                    reporter.game_completed(
                        game_b_id, res2["white"], res2["black"], res2["winner"],
                        res2.get("n_moves", 0), elo_info=elo_info2, error=res2.get("error")
                    )
                except Exception as e:
                    reporter.log_message(f"Error during match {game_b_id}: {e}")
                    time.sleep(1)
                    continue

        threads = []
        for w_id in range(jobs):
            t = threading.Thread(target=worker_loop, args=(w_id + 1,), daemon=True)
            t.start()
            threads.append(t)

        try:
            while any(t.is_alive() for t in threads):
                time.sleep(0.2)
        except KeyboardInterrupt:
            reporter.log_message("\nStopping arena (interrupted by user)...")
            stop_event.set()
        finally:
            stop_event.set()
            for t in threads:
                t.join(timeout=2)
            reporter.finish()


def load_engines_with_names(engine_paths_file="logs/paths.txt"):
    engine_paths = []
    names = []
    with open(engine_paths_file) as f:
        for path in f:
            path = path.strip()
            if not path or path.startswith("#"):
                continue
            engine_paths.append(path.replace('\\', '/'))
            names.append(path_to_name(engine_paths[-1]))
    return engine_paths, names


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hive bot arena')
    parser.add_argument("--engine_paths", type=str, default="logs/paths.txt",
                        help="File containing paths to the engine executables")
    parser.add_argument("-j", "--jobs", type=int, default=1,
                        help="Number of games to run in parallel (default: 1)")
    parser.add_argument("--update-elo", action=argparse.BooleanOptionalAction, default=True,
                        help="If the elo gets updated (default: True)")
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=True,
                        help="Display info about matches in real time (default: True)")
    parser.add_argument("-q", "--quiet", action="store_false", dest="verbose",
                        help="Disable verbose output (equivalent to --no-verbose)")
    parser.add_argument("--results-folder", type=str, default="logs/",
                        help="Folder where the matches are stored")
    parser.add_argument("--maxmoves", type=int, default=maxmoves,
                        help="Maximum number of moves per match per engine")
    parser.add_argument("--random-moves", type=int, default=0,
                        help="Number of initial random moves")

    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--timeout", type=float, default=5, help="Time per move")
    group.add_argument("--depth", type=int, default=0, help="Depth for engine evaluation (not used in arena)")
    args = parser.parse_args()

    if args.jobs < 1:
        parser.error("-j / --jobs must be at least 1")

    engine_paths, names = load_engines_with_names(args.engine_paths)

    # Check whether two different engines have the same path or name
    n_engines = len(engine_paths)
    for i in range(n_engines):
        for j in range(i + 1, n_engines):
            name_1, path_1 = names[i], engine_paths[i]
            name_2, path_2 = names[j], engine_paths[j]
            if path_1 == path_2:
                raise ValueError(f"Two different engines at the same location: \"{path_1}\"")
            elif name_1 == name_2:
                raise ValueError(
                    f"Names of different engines are the same:\n"
                    f" Name of engine at location \"{path_1}\" is {name_1}\n"
                    f" Name of engine at location \"{path_2}\" is {name_2}"
                )

    arena = HiveArena(engine_paths, results_folder=args.results_folder, jobs=args.jobs)
    arena.continuous_matches(
        engine_paths_file=args.engine_paths,
        update_elo=args.update_elo,
        verbose=args.verbose,
        timeout=args.timeout,
        depth=args.depth,
        maxmoves=args.maxmoves,
        random_moves=args.random_moves,
        jobs=args.jobs
    )
