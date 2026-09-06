import fastf1
import pandas as pd

fastf1.Cache.enable_cache('cache')

all_rows = []

years = [2022, 2023, 2024]

for year in years:
    schedule = fastf1.get_event_schedule(year)
    races = schedule[schedule['EventFormat'] != 'testing']

    for _, race in races.iterrows():
        round_number = race['RoundNumber']
        race_name = race['EventName']

        try:
            # Get qualifying results
            quali = fastf1.get_session(year, round_number, 'Q')
            quali.load(laps=False, telemetry=False, weather=False)
            quali_results = quali.results

            # Get race results
            race_session = fastf1.get_session(year, round_number, 'R')
            race_session.load(laps=False, telemetry=False, weather=False)
            race_results = race_session.results

            # Combine: for each driver, get their quali position and race position
            for _, driver_row in race_results.iterrows():
                driver_code = driver_row['Abbreviation']
                race_position = driver_row['Position']

                quali_row = quali_results[quali_results['Abbreviation'] == driver_code]
                if not quali_row.empty:
                    quali_position = quali_row['Position'].values[0]

                    all_rows.append({
                        'Year': year,
                        'Race': race_name,
                        'Driver': driver_code,
                        'QualiPosition': quali_position,
                        'RacePosition': race_position
                    })

            print(f"{year} {race_name}: done")

        except Exception as e:
            print(f"{year} {race_name}: could not load ({e})")

# Turn the collected data into a table and save it
dataset = pd.DataFrame(all_rows)
dataset.to_csv('f1_dataset.csv', index=False)
print(f"\nSaved {len(dataset)} rows to f1_dataset.csv")