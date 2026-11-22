"""Display and outcome definitions for independently scored football markets."""

# Each tuple is (stable key, display label, probability column, actual event key).
FEATURE_GROUPS = {
    "Full-time result (1X2)": [
        ("home_win", "Home win", "P_HOME", "HOME_WIN"),
        ("draw", "Draw", "P_DRAW", "DRAW"),
        ("away_win", "Away win", "P_AWAY", "AWAY_WIN"),
    ],
    "Double chance": [
        ("home_draw", "Home win or draw (1X)", "P_HOME_OR_DRAW", "HOME_DRAW"),
        ("draw_away", "Draw or away win (X2)", "P_DRAW_OR_AWAY", "DRAW_AWAY"),
        ("home_away", "Either team to win (12)", "P_HOME_OR_AWAY", "HOME_AWAY"),
    ],
    "Full-time goal totals": [
        ("over15", "Over 1.5 goals", "P_OVER15", "OVER15"),
        ("under15", "Under 1.5 goals", "P_UNDER15", "UNDER15"),
        ("over25", "Over 2.5 goals", "P_OVER25", "OVER25"),
        ("under25", "Under 2.5 goals", "P_UNDER25", "UNDER25"),
        ("over35", "Over 3.5 goals", "P_OVER35", "OVER35"),
        ("under35", "Under 3.5 goals", "P_UNDER35", "UNDER35"),
    ],
    "Both teams to score": [
        ("btts_yes", "Yes", "P_BTTS", "BTTS_YES"),
        ("btts_no", "No", "P_BTTS_NO", "BTTS_NO"),
    ],
    "Half-time result": [
        ("ht_home", "Home leading at half-time", "P_HT_HOME", "HT_HOME"),
        ("ht_draw", "Level at half-time", "P_HT_DRAW", "HT_DRAW"),
        ("ht_away", "Away leading at half-time", "P_HT_AWAY", "HT_AWAY"),
    ],
    "Half-time goals": [
        ("ht_goal", "1+ first-half goals", "P_FIRST_HALF_GOAL", "HT_GOAL"),
        ("ht_no_goal", "No first-half goal", "P_NO_FIRST_HALF_GOAL", "HT_NO_GOAL"),
        ("ht_2plus", "2+ first-half goals", "P_FIRST_HALF_2PLUS", "HT_2PLUS"),
        ("ht_under15", "Under 1.5 first-half goals", "P_FIRST_HALF_UNDER15", "HT_UNDER15"),
    ],
    "Corners (8.5 line)": [
        ("corners_over85", "Over 8.5 corners", "P_OVER85_CORNERS", "CORNERS_OVER85"),
        ("corners_under85", "Under 8.5 corners", "P_UNDER85_CORNERS", "CORNERS_UNDER85"),
    ],
    "Total shots (21.5 line)": [
        ("shots_over215", "Over 21.5 shots", "P_OVER215_SHOTS", "SHOTS_OVER215"),
        ("shots_under215", "Under 21.5 shots", "P_UNDER215_SHOTS", "SHOTS_UNDER215"),
    ],
    "Shots on target (7.5 line)": [
        ("sot_over75", "Over 7.5 shots on target", "P_OVER75_SHOTS_ON_TARGET", "SOT_OVER75"),
        ("sot_under75", "Under 7.5 shots on target", "P_UNDER75_SHOTS_ON_TARGET", "SOT_UNDER75"),
    ],
    "Yellow cards (3.5 line)": [
        ("yellow_over35", "Over 3.5 yellow cards", "P_OVER35_YELLOW_CARDS", "YELLOW_OVER35"),
        ("yellow_under35", "Under 3.5 yellow cards", "P_UNDER35_YELLOW_CARDS", "YELLOW_UNDER35"),
    ],
    "Red cards": [
        ("red_yes", "At least one red card", "P_ANY_RED_CARD", "RED_YES"),
        ("red_no", "No red card", "P_NO_RED_CARD", "RED_NO"),
    ],
    "Fouls (21.5 line)": [
        ("fouls_over215", "Over 21.5 fouls", "P_OVER215_FOULS", "FOULS_OVER215"),
        ("fouls_under215", "Under 21.5 fouls", "P_UNDER215_FOULS", "FOULS_UNDER215"),
    ],
    "Offsides (2.5 line)": [
        ("offsides_over25", "Over 2.5 offsides", "P_OVER25_OFFSIDES", "OFFSIDES_OVER25"),
        ("offsides_under25", "Under 2.5 offsides", "P_UNDER25_OFFSIDES", "OFFSIDES_UNDER25"),
    ],
}
