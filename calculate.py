import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os
import requests

########################################
# Professional Developer Debugging Note:
########################################
# The user wants to fetch temperature data using an API instead of using dummy/random data.
# Below, we replace the random-temperature logic with an example call to an external API.
# Keep in mind:
#   1) This environment might not have internet, so the code is for demonstration.
#   2) You need an API key to fetch real data.
#   3) APIs (like OpenWeatherMap) may require latitude/longitude or city name.
#   4) Historical data often requires specific endpoints or subscriptions.
#   5) If the user has a different API in mind, adapt accordingly.
#   6) We keep the rest of the code structure, test cases, etc.

########################
# Step 1: Load the CSV files with a check
########################

# NOTE: If these paths do not exist, the script will stop.
# Adjust them to your actual file paths.

electricity_path = './data/Verbruikshistoriek_elektriciteit_541448860018221378_20220308_20250309_dagtotalen.csv'
gas_path = './data/Verbruikshistoriek_gas_541448860018221361_20220308_20250309_dagtotalen.csv'

if not os.path.isfile(electricity_path):
    raise FileNotFoundError(f"Electricity file not found at: {electricity_path}\nPlease check your path or upload the file.")
if not os.path.isfile(gas_path):
    raise FileNotFoundError(f"Gas file not found at: {gas_path}\nPlease check your path or upload the file.")

# If we reach this point, both files exist.

elec_df = pd.read_csv(electricity_path, sep=';', decimal=',')
elec_df.rename(columns={"Van (datum)": "Date", "Volume": "Usage"}, inplace=True)


try:
    elec_df['Date'] = pd.to_datetime(elec_df['Date'], format='%d-%m-%Y', errors='raise')    
    elec_df.sort_values(by='Date', inplace=True)
except ValueError:
    print("\n[WARNING] Date parsing failed with '%d-%m-%Y'. Please check your actual date format.")
    raise

gas_df = pd.read_csv(gas_path, sep=';', decimal=',')
# Rename columns in the gas CSV to match the code references.
gas_df.rename(columns={"Van (datum)": "Date", "Volume": "Usage"}, inplace=True)

try:
    gas_df['Date'] = pd.to_datetime(gas_df['Date'], format='%d-%m-%Y', errors='raise')
    gas_df.sort_values(by='Date', inplace=True)
except ValueError:
    print("\n[WARNING] Date parsing failed for gas data with '%d-%m-%Y'. Please check your actual date format.")
    raise

########################
# Step 3: Fetch real temperature data from ERA5 reanalysis (Open-Meteo)
########################

import datetime

# Print debug info
earliest_elec = elec_df['Date'].min()
latest_elec = elec_df['Date'].max()
earl_gas = gas_df['Date'].min()
late_gas = gas_df['Date'].max()

print("Earliest electricity date in CSV:", earliest_elec)
print("Latest electricity date in CSV:\t", latest_elec)
print("Earliest gas date in CSV:\t", earl_gas)
print("Latest gas date in CSV:\t", late_gas)

# Use min and max of CSV data to set the query range
start_date = min(earliest_elec, earl_gas)
end_date = max(latest_elec, late_gas)

# Clamp end date to 'today' if needed
today_date = pd.to_datetime(datetime.date.today())
if end_date > today_date:
    end_date = today_date

# ERA5 data starts at 1979-01-01
min_coverage = pd.to_datetime("1979-01-01")
if start_date < min_coverage:
    start_date = min_coverage

start_date_str = start_date.strftime('%Y-%m-%d')
end_date_str = end_date.strftime('%Y-%m-%d')

print("\nAdjusted API date range:", start_date_str, "to", end_date_str)

LAT = 50.95  # Example latitude
LON = 3.12   # Example longitude

# Use the ERA5 archive endpoint
api_url = (
    f"https://archive-api.open-meteo.com/v1/era5?latitude={LAT}" 
    f"&longitude={LON}" 
    f"&start_date={start_date_str}" 
    f"&end_date={end_date_str}" 
    "&daily=temperature_2m_min,temperature_2m_max,temperature_2m_mean" 
    "&timezone=auto"
)

print("Fetching:", api_url)

try:
    response = requests.get(api_url, timeout=10)
    response.raise_for_status()
    weather_json = response.json()

    daily_data = weather_json.get('daily', {})
    dates_in_api = daily_data.get('time', [])
    temps_min_api = daily_data.get('temperature_2m_min', [])
    temps_max_api = daily_data.get('temperature_2m_max', [])
    temps_mean_api = daily_data.get('temperature_2m_mean', [])

    # Construct a DataFrame with columns "Date", "Min_Temp_C", "Max_Temp_C", "Avg_Temp_C".
    if (len(dates_in_api) == len(temps_min_api) == len(temps_max_api) == len(temps_mean_api)):
        temp_df = pd.DataFrame({
            'Date': pd.to_datetime(dates_in_api),
            'Min_Temp_C': temps_min_api,
            'Max_Temp_C': temps_max_api,
            'Avg_Temp_C': temps_mean_api
        })
        print("Successfully fetched", len(temp_df), "days of temperature data from ERA5 reanalysis.")
        print("Earliest temp date:", temp_df['Date'].min())
        print("Latest temp date:\t", temp_df['Date'].max())
    else:
        print("[WARNING] Mismatch in daily arrays.")
        temp_df = pd.DataFrame(columns=["Date", "Min_Temp_C", "Max_Temp_C", "Avg_Temp_C"])

except requests.exceptions.RequestException as e:
    print(f"[ERROR] Failed to fetch data from Open-Meteo: {e}")
    temp_df = pd.DataFrame(columns=["Date", "Min_Temp_C", "Max_Temp_C", "Avg_Temp_C"])

# Merge temperature with electricity usage.
merged_elec = pd.merge(elec_df, temp_df, on='Date', how='left')
merged_gas = pd.merge(gas_df, temp_df, on='Date', how='left')

########################
# Step 4: Correlation analysis
########################

# Ensure we have the usage and temperature columns.
if 'Usage' not in merged_elec.columns:
    raise KeyError("Column 'Usage' not found in electricity data. Please rename or update code.")
if 'Avg_Temp_C' not in merged_elec.columns:
    raise KeyError("Column 'Avg_Temp_C' not found in electricity data. Check your temperature fetch.")

if 'Usage' not in merged_gas.columns:
    raise KeyError("Column 'Usage' not found in gas data. Please rename or update code.")
if 'Avg_Temp_C' not in merged_gas.columns:
    raise KeyError("Column 'Avg_Temp_C' not found in gas data. Check your temperature fetch.")

correlation_elec = merged_elec['Usage'].corr(merged_elec['Avg_Temp_C'])
correlation_gas = merged_gas['Usage'].corr(merged_gas['Avg_Temp_C'])

print(f"Correlation between electricity usage and temperature: {correlation_elec:.2f}")
print(f"Correlation between gas usage and temperature: {correlation_gas:.2f}")

########################
# Step 6: Checking year-over-year usage changes (on a daily basis)
########################

# Instead of grouping by Month, we do day-of-year for each year.

# 1) Add "Year" and "DayOfYear" columns
merged_elec['Year'] = merged_elec['Date'].dt.year
merged_elec['DayOfYear'] = merged_elec['Date'].dt.dayofyear

merged_gas['Year'] = merged_gas['Date'].dt.year
merged_gas['DayOfYear'] = merged_gas['Date'].dt.dayofyear

# 2) Sum daily usage (if there are multiple readings per day, otherwise just use .mean())
daily_elec = merged_elec.groupby(['Year','DayOfYear'])['Usage'].sum().reset_index()
daily_elec.rename(columns={'Usage': 'Daily_Elec_Usage'}, inplace=True)

daily_gas = merged_gas.groupby(['Year','DayOfYear'])['Usage'].sum().reset_index()
daily_gas.rename(columns={'Usage': 'Daily_Gas_Usage'}, inplace=True)

print("\nDaily electricity usage sample:\n", daily_elec.head())
print("\nDaily gas usage sample:\n", daily_gas.head())

########################
# Step 7: Interpretation of results
########################
# Large changes not correlated with temperature might be due to new appliances, lifestyle changes, etc.

########################
# Basic test cases (ALWAYS add more test cases if none exist)
########################

print("\nTEST CASES:")
# 1) Are the DataFrames non-empty?
assert not elec_df.empty, "Electricity DataFrame is empty. Check file contents."
assert not gas_df.empty, "Gas DataFrame is empty. Check file contents."
print(" - PASS: DataFrames are not empty")

# 2) Check correlation calculation doesn't fail
assert correlation_elec == correlation_elec, "Electricity correlation is NaN"
assert correlation_gas == correlation_gas, "Gas correlation is NaN"
print(" - PASS: Correlation successfully calculated")

########################
# Step 8: Plot daily standard consumption and deviations by year
########################

# 1) Pivot daily electricity and gas usage from step 6
#    index = DayOfYear, columns = Year, values = usage
import matplotlib.pyplot as plt

daily_elec_pivot = daily_elec.pivot(index='DayOfYear', columns='Year', values='Daily_Elec_Usage')
daily_gas_pivot = daily_gas.pivot(index='DayOfYear', columns='Year', values='Daily_Gas_Usage')

# 2) Compute "standard" (average) daily usage across all years (for each day-of-year)
mean_elec_per_day = daily_elec_pivot.mean(axis=1)
mean_gas_per_day = daily_gas_pivot.mean(axis=1)

# 3) Plot lines for each year + the standard line for Electricity
daily_elec_pivot.plot(marker='o', linestyle='-')
plt.plot(mean_elec_per_day.index, mean_elec_per_day.values, linestyle='--', linewidth=3, label='Standard')
plt.xlabel('Day of Year (1=Jan1, 365=Dec31)')
plt.ylabel('Electricity Usage (kWh/day)')
plt.title('Daily Electricity Usage by Year vs. Standard')
plt.legend()
plt.show()

# Plot lines for each year + the standard line for Gas
daily_gas_pivot.plot(marker='o', linestyle='-')
plt.plot(mean_gas_per_day.index, mean_gas_per_day.values, linestyle='--', linewidth=3, label='Standard')
plt.xlabel('Day of Year (1=Jan1, 365=Dec31)')
plt.ylabel('Gas Usage (daily)')
plt.title('Daily Gas Usage by Year vs. Standard')
plt.legend()
plt.show()

# 4) Plot deviations from the standard: (usage - mean)

# Electricity deviations
dev_elec_daily = daily_elec_pivot.apply(lambda col: col - mean_elec_per_day)
dev_elec_daily.plot(marker='o', linestyle='-')
plt.xlabel('Day of Year')
plt.ylabel('Deviation (kWh/day) from Standard')
plt.title('Daily Electricity Usage Deviation by Year')
plt.show()

# Gas deviations
dev_gas_daily = daily_gas_pivot.apply(lambda col: col - mean_gas_per_day)
dev_gas_daily.plot(marker='o', linestyle='-')
plt.xlabel('Day of Year')
plt.ylabel('Deviation (Gas/day) from Standard')
plt.title('Daily Gas Usage Deviation by Year')
plt.show()

# 5) Consider temperature daily: group by Year, DayOfYear
# (We already have merged_elec with daily records. So we do a groupby similarly.)
daily_temp = merged_elec.groupby(['Year','DayOfYear'])['Avg_Temp_C'].mean().reset_index()

daily_temp_pivot = daily_temp.pivot(index='DayOfYear', columns='Year', values='Avg_Temp_C')

daily_temp_pivot.plot(marker='o', linestyle='-')
plt.xlabel('Day of Year')
plt.ylabel('Average Temperature (°C)')
plt.title('Daily Average Temperature by Year')
plt.show()

########################
# Step 9: Highlight excessive gas usage vs temperature differences (Daily)
########################

# 1) Compute the standard temperature for each day-of-year (avg across all years)
std_temp_day = daily_temp_pivot.mean(axis=1)

temp_dev_daily = daily_temp_pivot.apply(lambda col: col - std_temp_day)

# 2) We already have dev_gas_daily (gas usage deviation). Combine them.
temp_dev_stacked = temp_dev_daily.stack().reset_index()
temp_dev_stacked.columns = ['DayOfYear','Year','Temp_Dev']

dev_gas_stacked = dev_gas_daily.stack().reset_index()
dev_gas_stacked.columns = ['DayOfYear','Year','Gas_Dev']

# Merge on (DayOfYear, Year)
merged_dev_daily = pd.merge(dev_gas_stacked, temp_dev_stacked, on=['DayOfYear','Year'], how='inner')

# 3) Scatter plot of Gas_Dev vs Temp_Dev (daily)
plt.scatter(merged_dev_daily['Temp_Dev'], merged_dev_daily['Gas_Dev'])
plt.axvline(0, color='grey', linewidth=1)
plt.axhline(0, color='grey', linewidth=1)
plt.xlabel('Daily Temperature Deviation (°C from standard)')
plt.ylabel('Daily Gas Usage Deviation (from standard)')
plt.title('Daily Gas Usage Deviation vs. Temperature Deviation')
plt.show()

# 4) Another approach: highlight days that exceed a threshold
HIGH_USAGE_THRESHOLD = 2.0  # e.g., 2 kWh above standard
LOW_TEMP_THRESHOLD = -1.0   # 1°C below standard

high_usage_warm_days = merged_dev_daily[(merged_dev_daily['Gas_Dev'] > HIGH_USAGE_THRESHOLD) & (merged_dev_daily['Temp_Dev'] > 0)]
print("\nDays with significantly higher gas usage despite warmer temps:\n", high_usage_warm_days)
