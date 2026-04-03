import json
import subprocess
from pathlib import Path
import chess.pgn
from arena import load_engines_with_names
import sys

engine_paths, names = load_engines_with_names()
name_to_stem = {n: Path(p).stem for p, n in zip(engine_paths, names)}
target = "nokamute 1.0.3-2-g806ba76"

def full_name(name):
	stem = name_to_stem.get(name)
	name = name.replace("-dirty", "")
	return f"{name} ({stem})" if stem else name

with open('logs/results.json') as f, open('/tmp/games.pgn', 'w') as out:
	for r in json.load(f):
		if r["winner"] == "other": continue
		if r.get("error", None): continue
		g = chess.pgn.Game()
		for color in ('White', 'Black'):
			name = r[color.lower()]
			g.headers[color] = full_name(name)
		g.headers["Result"] = {"white":"1-0","black":"0-1"}.get(r["winner"],"1/2-1/2")
		g.headers["Date"] = r["datetime"].split("T")[0]
		g.comment = r["final_gamestate"]
		out.write(str(g) + "\n")

# https://github.com/michiguel/Ordo
args = ["ordo","-p","/tmp/games.pgn","-a","0","-W","-D"] + sys.argv[1:]
if target in names:
	args += ["-A", full_name(target)]
subprocess.run(args)
