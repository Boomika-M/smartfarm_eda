import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# CHANGE 1: show all columns (no "..." in the middle)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

df = pd.read_csv("uncleaned_smartfarm_dataset.csv")

#1 Exploring the data
print(df.head(5))
print("*" * 50)
print(df.tail(5))
print("*" * 50)
print(df.info())
print("*" * 50)
print("Numerical statistics")
print("____________________")
print(df.describe())
print("*" * 50)
print("Categorical statistics")
print("______________________")
# CHANGE 2: add "str" so the pandas warning disappears
print(df.describe(include=["object", "str"]))
print("*" * 50)
print("Shape of the dataset: ", df.shape)
print("*" * 50)
print("Columns in the dataset: ", df.columns)
print("*" * 50)

#2 Missing values and duplicates
print("Missing values")
print(df.isnull().sum())
print("*" * 50)
print("Duplicate rows: ", df.duplicated().sum())
print("*" * 50)

#3 Removing unappropriate rows and duplicate rows & unnecessary columns
a = df["reading_id"].isna() | df["reading_id"].astype(str).str.startswith("#") | (df["reading_id"] == "reading_id")
print("Number of rows to be removed: ", a.sum())
df = df[~a]
df = df.drop_duplicates()
df = df.drop(columns=["timezone_label", "gateway_id", "unnamed: 0"])

#4 Converting the data types of the columns
cols = ["latitude", "longitude", "air_temp", "humidity", "soil_moisture", "soil_ph",
        "rainfall_mm", "wind_speed", "battery_pct", "final_yield_kg_ha"]
for i in cols:
    df[i] = pd.to_numeric(df[i].astype(str).str.replace("%", ""), errors="coerce")

#5 Changing hidden missing values to NaN
df = df.replace(-999, np.nan)

#6 Changing the timestamp column to datetime
# CHANGE 3: the file has 3 time formats, so read each one in its own way
t = df["timestamp"].astype(str)
is_iso = t.str.contains("T")                                              # 2025-10-05T08:00:00+05:30
is_epoch = t.str.fullmatch(r"\d{9,11}")                                   # 1759650000
is_dmy = t.str.fullmatch(r"\d{2}-\d{2}-\d{4} \d{2}:\d{2} [AP]M")          # 05-10-2025 03:00 PM

t1 = pd.to_datetime(t[is_iso], errors="coerce", utc=True)
t2 = pd.to_datetime(t[is_epoch].astype("int64"), unit="s", utc=True)
t3 = (pd.to_datetime(t[is_dmy], format="%d-%m-%Y %I:%M %p", errors="coerce")
      .dt.tz_localize("Asia/Kolkata").dt.tz_convert("UTC"))
df["timestamp"] = pd.concat([t1, t2, t3]).reindex(df.index)

print("Bad timestamps: ", df["timestamp"].isna().sum())
# keep only October 2025 (this also removes the year 2035 rows)
df = df[df["timestamp"].between(pd.Timestamp("2025-09-30", tz="UTC"), pd.Timestamp("2025-11-01", tz="UTC"))]

# CHANGE 4: same station + same time twice = duplicate reading, keep one
df = df.sort_values(["station_id", "timestamp"])
df = df.drop_duplicates(subset=["station_id", "timestamp"])

#7 Making district and crop names like one district as one district and one crop
for i in ["district", "crop"]:
    df[i] = df.groupby("station_id")[i].transform(lambda x: x.mode()[0])
print(df["district"].value_counts())
print("*" * 50)

#8 Removing impossible values from the dataset
df.loc[df["humidity"] > 100, "humidity"] = np.nan
df.loc[df["rainfall_mm"] < 0, "rainfall_mm"] = np.nan
df.loc[~df["soil_ph"].between(3, 10), "soil_ph"] = np.nan
df.loc[~df["battery_pct"].between(0, 100), "battery_pct"] = np.nan

# CHANGE 5: two more checks
df.loc[~df["soil_moisture"].between(0, 100), "soil_moisture"] = np.nan
df.loc[df["wind_speed"] < 0, "wind_speed"] = np.nan

#9 Fixing units of whole stations (the boxplot showed 2 stations were different)
# CHANGE 6: look at the middle value of each station
by_station = df.groupby("station_id").agg(temp=("air_temp", "median"), wind=("wind_speed", "median"), rain=("rainfall_mm", "mean"))
print(by_station.round(2))
print("*" * 50)

f_stations = by_station.index[by_station["temp"] > 50]      # Fahrenheit
k_stations = by_station.index[by_station["wind"] > 8]       # km/h
i_stations = by_station.index[by_station["rain"] < 0.1]     # inches
print("Fahrenheit:", list(f_stations), "| km/h:", list(k_stations), "| inches:", list(i_stations))

df.loc[df["station_id"].isin(f_stations), "air_temp"] = (df["air_temp"] - 32) / 1.8
df.loc[df["station_id"].isin(k_stations), "wind_speed"] = df["wind_speed"] / 3.6
df.loc[df["station_id"].isin(i_stations), "rainfall_mm"] = df["rainfall_mm"] * 25.4

# temperature spikes like 85 or 999 are not real
df.loc[~df["air_temp"].between(-5, 42), "air_temp"] = np.nan

#10 Charts
sns.histplot(df["soil_moisture"].dropna(), kde=True)
plt.title("Soil Moisture")
plt.show()
sns.boxplot(df, x="station_id", y="air_temp")
plt.xticks(rotation=90)
plt.title("Temperature by station (after fixing units)")
plt.show()
sns.heatmap(df[["air_temp", "humidity", "soil_moisture", "rainfall_mm", "wind_speed"]].corr(), annot=True, cmap="coolwarm")
plt.title("Correlation between numerical features")
plt.show()

#11 Saving the cleaned dataset
df.to_csv("cleaned_smartfarm_dataset.csv", index=False)
print("Saved: ", df.shape)

# Solution graph: does rain increase soil moisture? (Thanjavur station)
one = df[df["station_id"] == "TN-S01"].copy()
one["date"] = one["timestamp"].dt.tz_convert("Asia/Kolkata").dt.date
daily = one.groupby("date").agg(rain=("rainfall_mm", "sum"), moisture=("soil_moisture", "mean"))

fig, ax1 = plt.subplots(figsize=(10, 5))
ax1.bar(daily.index, daily["rain"], color="skyblue")
ax1.set_ylabel("Daily rainfall (mm)")
ax2 = ax1.twinx()
ax2.plot(daily.index, daily["moisture"], color="darkgreen", marker="o")
ax2.set_ylabel("Average soil moisture")
plt.title("Thanjavur: soil moisture rises after rain")
fig.autofmt_xdate()
plt.show()