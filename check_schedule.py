import fastf1

fastf1.Cache.enable_cache('cache')

schedule = fastf1.get_event_schedule(2026)
print(schedule[['RoundNumber', 'EventName', 'EventDate']].to_string())