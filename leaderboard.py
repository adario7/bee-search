import pandas as pd

df = pd.read_json('logs/results.json')
valid = df[df.winner != 'other']

white = valid.assign(
    engine=valid.white,
    points=valid.winner.map({'white': 1, 'draw': 0.5}).fillna(0)
)
black = valid.assign(
    engine=valid.black,
    points=valid.winner.map({'black': 1, 'draw': 0.5}).fillna(0)
)

leaderboard = (
    pd.concat([white[['engine','points']], black[['engine','points']]])
      .groupby('engine')
      .agg(wins=('points','sum'), games=('engine','size'))
      .sort_values(['wins','games'], ascending=False)
)

print(leaderboard)
