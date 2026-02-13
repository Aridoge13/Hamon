import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# =========================================================
# CONFIGURATION
# =========================================================
NUM_USERS = 3000
DAYS = 30
START_DATE = datetime(2024, 1, 1)
OUTPUT_PATH = "/home/jijo/Projects/Hamon/data/hamon_googlefit_medical.csv"

rng = np.random.default_rng(42)


# USER-LEVEL PROFILES (STATIC)
user_ids = np.arange(1, NUM_USERS + 1)

age = rng.integers(18, 80, NUM_USERS)
sex = rng.integers(0, 2, NUM_USERS)  # 0=female, 1=male
bmi = rng.normal(26, 4.5, NUM_USERS).clip(18, 42)

smoking = rng.choice([0, 1], NUM_USERS, p=[0.8, 0.2])
family_history = rng.choice([0, 1], NUM_USERS, p=[0.7, 0.3])
fitness_level = rng.choice([0, 1, 2], NUM_USERS, p=[0.4, 0.4, 0.2])

height = rng.normal(1.70, 0.1, NUM_USERS).clip(1.45, 2.0)
baseline_weight = bmi * (height ** 2)

baseline_rhr = rng.normal(68, 8, NUM_USERS) - fitness_level * 5
baseline_hrv = rng.normal(45, 12, NUM_USERS) + fitness_level * 10

baseline_risk = (
    (age - 18) / 62
    + smoking * 0.6
    + family_history * 0.5
    + (bmi - 22) / 15
)


# ACTIVITY TYPES (SIMPLIFIED REALISTIC)
activity_types = [
    "Rest",
    "Walking",
    "Running",
    "Cycling",
    "Strength",
    "Mixed_Cardio",
    "Yoga",
]

activity_probs = [0.15, 0.35, 0.15, 0.10, 0.10, 0.10, 0.05]


# DATA GENERATION
records = []

for i in range(NUM_USERS):

    weight = baseline_weight[i]

    for d in range(DAYS):

        date = START_DATE + timedelta(days=d)

        # Latent daily states
        autonomic_load = rng.normal(0, 1)
        metabolic_stress = rng.normal(0, 1)
        recovery = rng.normal(0, 1) + fitness_level[i] * 0.5

        activity = rng.choice(activity_types, p=activity_probs)

        # Activity & Movement
        base_steps = {
            "Rest": 1500,
            "Walking": 7000,
            "Running": 11000,
            "Cycling": 9000,
            "Strength": 5000,
            "Mixed_Cardio": 10000,
            "Yoga": 4000,
        }[activity]

        steps = int(np.clip(rng.normal(base_steps + recovery * 1200, 2000), 0, 30000))
        move_minutes = int((steps / 1000) * rng.uniform(5, 9))
        distance_km = round(steps * 0.00075, 2)

        heart_points = int(np.clip(move_minutes * rng.uniform(0.6, 1.4), 0, 120))


        calories_burned = int(
            1600
            + steps * 0.04
            + fitness_level[i] * 100
            + rng.normal(0, 120)
        )

        step_cadence = round(rng.normal(85, 10), 1) if steps > 1000 else 0
        cycling_power = (
            round(rng.normal(150, 40), 1) if activity == "Cycling" else 0
        )

        # Sleep
        sleep_available = rng.random() > 0.3
        sleep_hours = np.clip(rng.normal(7.2 - autonomic_load * 0.6, 1.1), 3, 10)
        sleep_eff = np.clip(rng.normal(0.85 - autonomic_load * 0.05, 0.06), 0.5, 0.98)


        if not sleep_available:
            sleep_hours = np.nan
            sleep_eff = np.nan

        # Cardiovascular
        rhr = baseline_rhr[i] + autonomic_load * 4 - recovery * 2
        hrv = baseline_hrv[i] - autonomic_load * 6 + recovery * 4

        rhr = int(np.clip(rhr, 45, 100))
        hrv = int(np.clip(hrv, 10, 140))

        avg_hr = int(rhr + move_minutes * 0.3)

        # Blood Pressure
        bp_measured = rng.random() < 0.5

        if bp_measured:
            bp_sys = int(
                110
                + baseline_risk[i] * 20
                + autonomic_load * 6
                + metabolic_stress * 4
                + rng.normal(0, 5)
            )
            bp_dia = int(bp_sys * rng.uniform(0.6, 0.7))
        else:
            bp_sys = np.nan
            bp_dia = np.nan

        # Glucose
        glucose_measured = rng.random() < 0.3

        if glucose_measured:
            fasting_glucose = int(
                90
                + metabolic_stress * 8
                + baseline_risk[i] * 10
                + rng.normal(0, 8)
            )
            post_glucose = int(fasting_glucose + rng.uniform(15, 60))
        else:
            fasting_glucose = np.nan
            post_glucose = np.nan

        # SpO2 & Temperature
        spo2 = int(np.clip(rng.normal(97 - baseline_risk[i], 1), 88, 100))
        body_temp = round(rng.normal(36.6, 0.25), 1)

        # Fatigue
        fatigue = int(
            np.clip(
                5 + autonomic_load * 2 - recovery * 1.5 + rng.normal(0, 1),
                0,
                10,
            )
        )

        # Weight Drift
        weight += rng.normal(0, 0.05)
        weight = float(np.clip(weight, 45, 160))

        # Nutrition
        calories_in = int(np.clip(rng.normal(2200 + bmi[i] * 10, 350), 1200, 4500))
        water_intake = round(np.clip(rng.normal(2.3, 0.7), 0.5, 6), 1)


        # Risk Score
        risk_score = (
            baseline_risk[i] * 0.5
            + autonomic_load * 0.2
            + metabolic_stress * 0.2
            + (rhr - baseline_rhr[i]) / 20
            - recovery * 0.2
        )

        risk_state = int(np.clip(risk_score + 2, 0, 4))

        # Record
        records.append([
            user_ids[i],
            age[i],
            sex[i],
            bmi[i],
            smoking[i],
            family_history[i],
            fitness_level[i],
            height[i],
            weight,
            date,
            d + 1,
            activity,
            steps,
            heart_points,
            move_minutes,
            calories_burned,
            distance_km,
            step_cadence,
            cycling_power,
            avg_hr,
            rhr,
            hrv,
            bp_sys,
            bp_dia,
            fasting_glucose,
            post_glucose,
            spo2,
            body_temp,
            sleep_hours,
            sleep_eff,
            fatigue,
            calories_in,
            water_intake,
            sleep_available,
            bp_measured,
            glucose_measured,
            risk_state,
        ])


# CREATE DATAFRAME
columns = [
    "user_id",
    "age",
    "sex",
    "bmi",
    "smoking_status",
    "family_history_cvd",
    "fitness_level",
    "height_m",
    "weight_kg",
    "date",
    "day_index",
    "activity_type",
    "steps",
    "heart_points",
    "move_minutes",
    "calories_burned",
    "distance_km",
    "step_cadence",
    "cycling_power_watts",
    "avg_heart_rate",
    "resting_hr",
    "hrv",
    "bp_systolic",
    "bp_diastolic",
    "fasting_glucose",
    "postprandial_glucose",
    "spo2",
    "body_temp_c",
    "sleep_hours",
    "sleep_efficiency",
    "fatigue_score",
    "calories_consumed",
    "water_intake_l",
    "sleep_data_available",
    "bp_measured",
    "glucose_measured",
    "cardiometabolic_risk_state",
]

df = pd.DataFrame(records, columns=columns)

# SAVE FILE
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
df.to_csv(OUTPUT_PATH, index=False)

print(f"Saved {len(df)} rows with {len(df.columns)} columns.")
