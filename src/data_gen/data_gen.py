import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os


# CONFIG
NUM_USERS = 3000
DAYS = 30
START_DATE = datetime(2024, 1, 1)
OUTPUT_PATH = "/home/jijo/Projects/Hamon/data/hamon_googlefit_medical_realistic.csv"

rng = np.random.default_rng(42)

# USER PROFILES (STATIC)
user_ids = np.arange(1, NUM_USERS + 1)

age = rng.integers(18, 80, NUM_USERS)
sex = rng.integers(0, 2, NUM_USERS)
bmi = rng.normal(26, 4.5, NUM_USERS).clip(18, 42)

smoking = rng.choice([0, 1], NUM_USERS, p=[0.8, 0.2])
family_history = rng.choice([0, 1], NUM_USERS, p=[0.7, 0.3])
fitness = rng.choice([0, 1, 2], NUM_USERS, p=[0.4, 0.4, 0.2])

height = rng.normal(1.70, 0.1, NUM_USERS).clip(1.45, 2.0)
weight0 = bmi * (height ** 2)

rhr_base = rng.normal(68, 8, NUM_USERS) - fitness * 5
hrv_base = rng.normal(45, 12, NUM_USERS) + fitness * 10

baseline_risk = (
    (age - 18) / 62
    + smoking * 0.6
    + family_history * 0.5
    + (bmi - 22) / 15
)

activity_types = ["Rest","Walking","Running","Cycling","Strength","Mixed_Cardio","Yoga"]

records = []

# SIMULATION PER USER
for i in range(NUM_USERS):

    weight = weight0[i]

    # Persistent latent states
    autonomic = rng.normal(0, 0.5)
    metabolic = rng.normal(0, 0.5)
    recovery = fitness[i] * 0.5

    illness_days = set()

    # Occasional illness episode
    if rng.random() < 0.25:
        start = rng.integers(0, DAYS - 5)
        length = rng.integers(3, 7)
        illness_days = set(range(start, start + length))

    for d in range(DAYS):

        date = START_DATE + timedelta(days=d)
        weekend = date.weekday() >= 5

        
        # TEMPORAL STATE EVOLUTION
        autonomic = 0.85 * autonomic + rng.normal(0, 0.3)
        metabolic = 0.90 * metabolic + rng.normal(0, 0.2)
        recovery = 0.80 * recovery + fitness[i] * 0.3 + rng.normal(0, 0.2)

        if d in illness_days:
            autonomic += 2.5
            metabolic += 1.5
            recovery -= 2

        
        # SLEEP
        sleep_available = rng.random() > 0.25

        sleep_hours = np.clip(
            7.2 - autonomic * 0.6 + weekend * 0.5 + rng.normal(0, 0.7),
            3,
            10,
        )

        sleep_eff = np.clip(0.88 - autonomic * 0.05 + rng.normal(0, 0.05), 0.5, 0.98)

        if not sleep_available:
            sleep_hours = np.nan
            sleep_eff = np.nan

        
        # RECOVERY / FATIGUE
        fatigue = int(np.clip(5 + autonomic * 1.5 - recovery + rng.normal(0, 1), 0, 10))

        # ACTIVITY
        if d in illness_days:
            activity = "Rest"
        else:
            activity = rng.choice(activity_types)

        base_steps = {
            "Rest": 2000,
            "Walking": 7000,
            "Running": 11000,
            "Cycling": 9000,
            "Strength": 5000,
            "Mixed_Cardio": 10000,
            "Yoga": 4000,
        }[activity]

        steps = int(np.clip(base_steps + recovery * 1200 - fatigue * 300 + rng.normal(0, 1500), 0, 30000))

        move_minutes = int(steps / 1000 * rng.uniform(5, 8))
        distance_km = round(steps * 0.00075, 2)

        heart_points = int(np.clip(move_minutes * rng.uniform(0.7, 1.3), 0, 120))

        calories_burned = int(1600 + steps * 0.04 + rng.normal(0, 120))

        step_cadence = round(rng.normal(85, 8), 1) if steps > 1000 else 0
        cycling_power = round(rng.normal(150, 40), 1) if activity == "Cycling" else 0

        # CARDIOVASCULAR
        rhr = int(np.clip(rhr_base[i] + autonomic * 4 - recovery * 2, 45, 100))
        hrv = int(np.clip(hrv_base[i] - autonomic * 6 + recovery * 4, 10, 140))
        avg_hr = int(rhr + move_minutes * 0.3)

        # BP (measured intermittently)
        bp_measured = rng.random() < 0.4

        if bp_measured:
            bp_sys = int(np.clip(110 + baseline_risk[i] * 20 + autonomic * 6 + rng.normal(0, 5), 90, 200))
            bp_dia = int(bp_sys * rng.uniform(0.6, 0.7))
        else:
            bp_sys = np.nan
            bp_dia = np.nan

        # GLUCOSE
        glucose_measured = rng.random() < 0.25

        if glucose_measured:
            fasting_glucose = int(np.clip(90 + metabolic * 8 + baseline_risk[i] * 10 + rng.normal(0, 8), 70, 250))
            post_glucose = int(fasting_glucose + rng.uniform(15, 60))
        else:
            fasting_glucose = np.nan
            post_glucose = np.nan

        # OTHER VITALS
        spo2 = int(np.clip(rng.normal(97 - baseline_risk[i], 1), 88, 100))
        body_temp = round(36.6 + autonomic * 0.1 + rng.normal(0, 0.2), 1)

        # WEIGHT (slow drift)
        weight += rng.normal(0, 0.03)
        weight = float(np.clip(weight, 45, 160))

        calories_in = int(np.clip(rng.normal(2200 + bmi[i] * 10, 300), 1200, 4500))
        water_intake = round(np.clip(rng.normal(2.3, 0.6), 0.5, 6), 1)

        # RISK SCORE (smoothed)
        risk_cont = (
            baseline_risk[i] * 0.5
            + autonomic * 0.3
            + metabolic * 0.2
            + (rhr - rhr_base[i]) / 20
            - recovery * 0.2
        )

        risk_state = int(np.clip(risk_cont + 2, 0, 4))

        records.append([
            user_ids[i], age[i], sex[i], bmi[i], smoking[i],
            family_history[i], fitness[i], height[i], weight,
            date, d + 1, activity, steps, heart_points,
            move_minutes, calories_burned, distance_km,
            step_cadence, cycling_power, avg_hr, rhr, hrv,
            bp_sys, bp_dia, fasting_glucose, post_glucose,
            spo2, body_temp, sleep_hours, sleep_eff,
            fatigue, calories_in, water_intake,
            sleep_available, bp_measured,
            glucose_measured, risk_state
        ])

# DATAFRAME
columns = [
    "user_id","age","sex","bmi","smoking_status",
    "family_history_cvd","fitness_level","height_m",
    "weight_kg","date","day_index","activity_type",
    "steps","heart_points","move_minutes","calories_burned",
    "distance_km","step_cadence","cycling_power_watts",
    "avg_heart_rate","resting_hr","hrv",
    "bp_systolic","bp_diastolic","fasting_glucose",
    "postprandial_glucose","spo2","body_temp_c",
    "sleep_hours","sleep_efficiency","fatigue_score",
    "calories_consumed","water_intake_l",
    "sleep_data_available","bp_measured",
    "glucose_measured","cardiometabolic_risk_state",
]

df = pd.DataFrame(records, columns=columns)

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
df.to_csv(OUTPUT_PATH, index=False)

print(f"Saved {len(df)} rows with {len(df.columns)} columns.")
