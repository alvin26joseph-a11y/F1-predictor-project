# F1 Race Result Predictor

A machine learning project that predicts Formula 1 race results — finishing position and DNF risk — using historical qualifying and race data. Built from scratch in Python, and tested live against real, upcoming races.

## What this does

Using the [fastf1](https://github.com/theOehrly/Fast-F1) library, this project:

1. **Builds a historical dataset** spanning four full seasons (2022-2025) — qualifying position and race result for every driver, every race
2. **Trains a baseline model** predicting finishing position from qualifying position alone
3. **Adds recent form** (a driver's average finish over their last 3 races) as a second input, and shows how this changes the model's understanding of what qualifying position alone was really capturing
4. **Converts raw predictions into labels** — predicted race winner, podium, and top-10 finish — and checks how often those predictions are actually correct
5. **Tests the model on a real, live race** it had never seen (the 2026 Italian Grand Prix), comparing predictions to the actual result
6. **Adds a DNF risk model** — a separate classifier estimating each driver's probability of retiring from a race, based on their qualifying position and career reliability history
7. **Predicts upcoming races automatically** by pulling live qualifying data the moment it becomes available, with no manual data entry

## What I found

- **Qualifying position alone explains some, but not most, of a race result.** A driver who qualifies 10 places back is predicted to finish only about 6 places back on average — a real relationship, but with huge scatter around it.
- **Recent form and qualifying position overlap.** Adding recent form as a second input reduced qualifying position's apparent effect, because some of what qualifying position was "explaining" was really being driven by a driver's underlying current form.
- **The model reliably identifies the true favorite and the back of the field, but struggles in the middle.** Tested against real race outcomes, it correctly predicted race winners about 39% of the time (versus a ~5% random baseline) and podium finishers about 44% of the time (versus a ~15% random baseline) — but was weaker at pinpointing exact top-10 cutoffs, where the pack is most tightly bunched.
- **Real-world testing against the 2026 Italian Grand Prix** showed genuine strengths and a clear, honest limitation: the model nailed Hamilton's finishing position exactly and predicted Russell and Verstappen within one position each — but completely missed a surprise rookie win (regression to the mean: the model plays it safe and can't foresee breakout performances) and had no way to anticipate three DNFs, including one for a driver it had predicted to finish on the podium.
- **DNF risk is a real, learnable signal.** A driver's career-long retirement rate, combined with their qualifying position, gives a genuine probability estimate for whether they'll finish a given race — something the earlier position-only model had no way to account for.

## What I'd try next

- Incorporate practice session pace as an additional pre-qualifying signal
- Add weather data, since wet-race unpredictability likely explains some of the model's mid-pack weakness
- Try a non-linear model (e.g. random forest) to see if it captures driver/track interactions better than linear regression
- Track prediction accuracy across many races over a full season to get a more statistically reliable read on real-world performance, rather than judging from a single test race

## Project structure

```
build_dataset.py        - Builds/resumes the historical dataset (2022-2025)
predict_position.py     - Baseline and recent-form position models
check_race.py           - Inspects model predictions against a specific past race
predict_future_race.py  - Predicts a race using only pre-race information
predict_next_race.py    - Live prediction: pulls real qualifying data automatically,
                           includes DNF risk model
f1_dataset.csv           - The built historical dataset
f1_predictions.csv       - Enriched dataset with predictions and accuracy labels
```

## Tools used

Python, pandas, scikit-learn (LinearRegression, LogisticRegression), matplotlib, [fastf1](https://github.com/theOehrly/Fast-F1)

## How to run it

```
pip install fastf1 pandas scikit-learn matplotlib
python build_dataset.py       # builds the dataset (resumable if interrupted)
python predict_position.py    # trains and evaluates the position model
python predict_next_race.py   # predicts the next upcoming race live
```

The first dataset build takes a while due to the volume of historical data and occasional API rate limits — the script saves progress after every race and can simply be re-run to continue where it left off.
