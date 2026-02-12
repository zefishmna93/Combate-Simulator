import streamlit as st
import math
import random
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# -----------------------------
# Core math (same as before)
# -----------------------------

HEALTH_SCALAR = 5.0
HEALTH_EXP = 1.1
DAMAGE_SCALAR = 0.8
DAMAGE_EXP = 0.75
ACCURACY_MID = 80
ACCURACY_STEEPNESS = 0.03
MIN_ACCURACY = 0.15
MAX_ACCURACY = 0.95
WEIGHT_BIAS_RANGE = 60
WEIGHT_EXPONENT = 1.4
MIN_DAMAGE_FACTOR = 0.25
DMG_SCALE = 0.6
SIM_HITS = 5000

def total_power(level):
    return math.ceil(SCALE * (level ** EXPONENT))

def power_to_accuracy(acc_power):
    val = 1 / (1 + np.exp((-1 * ACCURACY_STEEPNESS) * (acc_power - ACCURACY_MID)))
    return max(MIN_ACCURACY, min(MAX_ACCURACY, val))

def roll_damage(max_damage, accuracy):
    r = random.random()
    acc_mod = r ** (1/accuracy)
    return max_damage * acc_mod

def enemy_stats(level, w):
    tp = total_power(level)
    return {
        "Health": round(((tp * w["Health Weight"]) ** HEALTH_EXP) * HEALTH_MULT),
        "MaxDamage": round(((tp * w["Damage Weight"]) ** DAMAGE_EXP) * DAMAGE_MULT),
        "Accuracy": round(power_to_accuracy(tp + (w["Accuracy Weight"] - 0.5) * WEIGHT_BIAS_RANGE), 2),
    }

def player_combat_level(a):
    level = (a["Raw Health"] + a["Raw Strength"] + a["Raw Attack"]) * 2 / 3

    return round(level)

def player_health(h):
    BASE_HP = 3
    EXP = 1.33

    return round(BASE_HP + h ** EXP)

def player_max_damage(s):
    BASE_DMG = 3

    return round(BASE_DMG + (s * DMG_SCALE))

def player_accuracy(a):
    BASE_ACC = MIN_ACCURACY
    LOG_SCALE = 0.173
    
    return min(MAX_ACCURACY, round(BASE_ACC + LOG_SCALE * np.log(a), 2))

def player_stats(a):
    return {
        "Health": player_health(a["Raw Health"]),
        "MaxDamage": player_max_damage(a["Raw Strength"]),
        "Accuracy": player_accuracy(a["Raw Attack"])
    }


def simulate(attacker):
    return [round(roll_damage(attacker["MaxDamage"], attacker["Accuracy"])) for _ in range(SIM_HITS)]
    
def simulate_fight(player, enemy, max_rounds=1000):
    player_hp = player["Health"]
    enemy_hp = enemy["Health"]

    player_max_dmg = player["MaxDamage"]
    player_acc = player["Accuracy"]

    enemy_max_dmg = enemy["MaxDamage"]
    enemy_acc = enemy["Accuracy"]

    for round_num in range(max_rounds):
        # Player Attacks
        dmg = roll_damage(player_max_dmg, player_acc)
        enemy_hp -= dmg

        if enemy_hp <= 0:
            return "Player", round_num + 1
        
        dmg = roll_damage(enemy_max_dmg, enemy_acc)
        player_hp -= dmg

        if player_hp <= 0:
            return "Enemy", round_num + 1

def simulate_matchup(player, enemy, simulations=5000):
    results = {
        "Player Wins": 0,
        "Enemy Wins": 0,
        "Rounds to Resolution": []
    }

    for _ in range(simulations):
        winner, rounds = simulate_fight(player, enemy)
        results["Rounds to Resolution"].append(rounds)

        if winner == "Player":
            results["Player Wins"] += 1
        elif winner == "Enemy":
            results["Enemy Wins"] += 1

    return results, simulations

def analyze_results(results, simulations):
    win_rates = {
        "Enemy Win %": results["Enemy Wins"] / simulations * 100,
        "Player Win %": results["Player Wins"] / simulations * 100,
    }

    rounds = np.array(results["Rounds to Resolution"])

    round_stats = {
        "Avg Rounds": rounds.mean(),
        "Median Rounds": np.median(rounds),
        "Min Rounds": rounds.min(),
        "Max Rounds": rounds.max()
    }

    return win_rates, round_stats

# -----------------------------
# Streamlit UI
# -----------------------------

st.title("Combat Balance Sandbox")

hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# -------------------------
# Sidebar Inputs
# -------------------------
st.sidebar.header("Global Scaling")

EXPONENT = st.sidebar.slider("Level Exponent", 0.8, 2.0, 1.25)
SCALE = st.sidebar.slider("Power Scale", 0.0, 1.0, 0.1)

HEALTH_MULT = st.sidebar.slider("Health Multiplier", 5, 100, 20)
HEALTH_EXP = st.sidebar.slider("Health Exponent", 1.0, 2.0, 1.1)
DAMAGE_MULT = st.sidebar.slider("Damage Multiplier", 1, 20, 5)
DAMAGE_EXP = st.sidebar.slider("Damage Exponent", 0.5, 1.0, 0.75)
ACCURACY_MID = st.sidebar.slider("Accuracy Midpoint", 0, 100, 80)
ACCURACY_STEEPNESS = st.sidebar.slider("Accuracy Steepness", 0.01, 0.3, 0.03)
WEIGHT_BIAS_RANGE = st.sidebar.slider("Accuracy Bias Range", 0, 100, 60)
DMG_SCALE = st.sidebar.slider("Player Damage Scalar", 0.1, 1.0, 0.6)

st.sidebar.header("Enemy Configuration")

PRESETS = {
    "Balanced": [0.33, 0.33, 0.34],
    "Tank": [0.55, 0.30, 0.15],
    "Bruiser": [0.45, 0.40, 0.15],
    "Sniper": [0.25, 0.35, 0.40],
    "Glass Cannon": [0.20, 0.50, 0.30]
}


enemy_level = st.sidebar.slider("Enemy Level", 1, 300, 20)

option = st.sidebar.selectbox("Enemy Preset", ("Balanced", "Tank", "Bruiser", "Sniper", "Glass Cannon"), accept_new_options=False)
weights = PRESETS[option]


eh = st.sidebar.slider("Health Weight", 0.0, 1.0, weights[0])
ed = st.sidebar.slider("Damage Weight", 0.0, 1.0, weights[1])
ea = st.sidebar.slider("Accuracy Weight", 0.0, 1.0, weights[2])

enemy_weights = {
    "Health Weight": eh,
    "Damage Weight": ed,
    "Accuracy Weight": ea
}

st.sidebar.header("Player Stats")
hel = st.sidebar.slider("Hel", 1, 100, 40)
str_ = st.sidebar.slider("Str", 1, 100, 40)
atk = st.sidebar.slider("Atk", 1, 100, 20)

player_stats_raw = {"Raw Health": hel, "Raw Strength": str_, "Raw Attack": atk}
# -------------------------
# Derived Values
# -------------------------



enemy = {"Level": enemy_level} | enemy_stats(enemy_level, enemy_weights)
player = {"Level": player_combat_level(player_stats_raw)} | player_stats(player_stats_raw)

df = pd.DataFrame.from_dict([enemy, player])
df.index=["Enemy", "Player"]
df = df.T

st.subheader("Derived Values")
st.dataframe(df)

# -------------------------
# Damage Distribution
# -------------------------

enemy = enemy_stats(enemy_level, enemy_weights)
player = player_stats(player_stats_raw)

enemy_dmg = simulate(enemy)
player_dmg = simulate(player)

st.subheader("Damage Distribution")
fig, ax = plt.subplots()
ax.hist(enemy_dmg, bins=50, alpha=0.5, label="Enemy")
ax.hist(player_dmg, bins=50, alpha=0.5, label="Player")
ax.legend()
st.pyplot(fig)

# -------------------------
# Statistics
# -------------------------

df = pd.DataFrame({
    "Enemy Damage": enemy_dmg,
    "Player Damage": player_dmg
})

st.subheader("Summary Statistics")
st.dataframe(df.describe())

results, simulations = simulate_matchup(player, enemy)

win_rates, round_stats = analyze_results(results, simulations)


df = pd.DataFrame.from_dict([win_rates])
df.index = ["Results"]
st.subheader("Win Rates")
st.dataframe(df)

df = pd.DataFrame.from_dict([round_stats])
df.index = ["Stats"]
st.subheader("Round Statistics")

st.dataframe(df)
