import fastf1
import pandas as pd
import os

fastf1.Cache.enable_cache('cache')

years = [2022, 2023, 2024, 2025]

# Load existing data if we have any from a previous run
if os.path.exists('f1_dataset.csv'):
    existing = pd.read_csv('f1_dataset.csv')
    done_races = set(zip(existing['Year'], existing['Race']))
    all_rows = existing.to_dict('records')
    print(f"Resuming - found {len(existing)} existing rows, {len(done_races)} races already done")
else:
    existing = pd.DataFrame()
    done_races = set()
    all_rows = []

for year in years:
    schedule = fastf1.get_event_schedule(year)
    races = schedule[schedule['EventFormat'] != 'testing']

    for _, race in races.iterrows():
        round_number = race['RoundNumber']
        race_name = race['EventName']

        # Skip races we've already fetched in a previous run
        if (year, race_name) in done_races:
            continue

        try:
            quali = fastf1.get_session(year, round_number, 'Q')
            quali.load(laps=False, telemetry=False, weather=False)
            quali_results = quali.results

            race_session = fastf1.get_session(year, round_number, 'R')
            race_session.load(laps=False, telemetry=False, weather=False)
            race_results = race_session.results

            new_rows = []
            for _, driver_row in race_results.iterrows():
                driver_code = driver_row['Abbreviation']
                race_position = driver_row['Position']

                quali_row = quali_results[quali_results['Abbreviation'] == driver_code]
                if not quali_row.empty:
                    quali_position = quali_row['Position'].values[0]
                    new_rows.append({
                        'Year': year,
                        'Race': race_name,
                        'Driver': driver_code,
                        'QualiPosition': quali_position,
                        'RacePosition': race_position
                    })

            all_rows.extend(new_rows)

            # Save after EVERY race, so a rate limit later doesn't lose this progress
            pd.DataFrame(all_rows).to_csv('f1_dataset.csv', index=False)
            print(f"{year} {race_name}: done and saved ({len(new_rows)} rows)")

        except Exception as e:
            print(f"{year} {race_name}: could not load ({e})")
            print("Stopping here - progress so far is saved. Re-run this script later to continue.")
            exit()

print(f"\nFinished! Total rows in f1_dataset.csv: {len(all_rows)}")