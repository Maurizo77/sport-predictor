from nba_api.live.nba.endpoints import scoreboard

board = scoreboard.ScoreBoard()
games = board.get_dict()["scoreboard"]["games"]

print("Games today:", len(games))
for game in games:
    home = game["homeTeam"]["teamTricode"]
    away = game["awayTeam"]["teamTricode"]
    print(away, "@", home, "-", game["gameStatusText"])