# F1 Race Result Predictor

A machine learning project that predicts Formula 1 race results — finishing position and DNF risk — using historical qualifying and race data. Built from scratch in Python, and validated live against real, upcoming races as they happen.

## What this does

Using the [fastf1](https://github.com/theOehrly/Fast-F1) library, this project:

1. **Builds a historical dataset** spanning four full seasons (2022-2025) — qualifying position and race result for every driver, every race — with a resumable pipeline that survives API rate limits.
2. **Trains a baseline model** predicting finishing position from qualifying position alone.
3. **Adds recent form** (a driver's average finish over their last 3 races) as a second input, and shows how this changes the model's understanding of what qualifying position alone was really capturing.
4. **Converts raw predictions into labels** — predicted race winner, podium, and top-10 finish — and checks how often those predictions are actually correct against historical results.
5. **Adds a DNF risk model** — a separate classifier estimating each driver's probability of retiring from a race, based on their qualifying position and career reliability history.
6. **Predicts upcoming races automatically** by pulling live qualifying data the moment it becomes available, with no manual data entry required.
7. **Is tested against real races as they happen** — not just backtested on historical data — with predictions compared to actual results and failures investigated rather than glossed over.

## What I found

- **Qualifying position alone explains some, but not most, of a race result.** A driver who qualifies 10 places back is predicted to finish only about 6 places back on average — a real relationship, but with a lot of scatter around it.
- **Recent form and qualifying position overlap.** Adding recent form as a second input reduced qualifying position's apparent effect, because some of what qualifying position was "explaining" was really being driven by a driver's underlying current form.
- **Backtested against historical results**, the model correctly identified race winners about 39% of the time (versus a ~5% random baseline) and podium finishers about 44% of the time (versus a ~15% random baseline) — strong at the front of the field, weaker at pinpointing the exact top-10 cutoff where the midfield is tightly bunched.

### Live test 1 — 2026 Italian Grand Prix
Predicted using only that morning's real qualifying data, for a race the model had never seen. It correctly placed Hamilton's finishing position exactly, and had Russell and Verstappen within one position each — but completely missed a surprise rookie win (a case of regression to the mean: the model plays it safe and cannot foresee breakout performances) and had no way to anticipate three DNFs, including one for a driver it had predicted to finish on the podium.

### Live test 2 — 2026 Spanish Grand Prix
A second live test, one week later. **Position predictions improved**: two exact matches (Leclerc, Russell), and the predicted top group (Verstappen, Norris, Antonelli) correctly identified as the front-runners — including Antonelli's predicted rank rising noticeably after his Monza win fed back into his recent-form value, a real example of the model adapting week to week as new results come in.

**The DNF model, however, was shown to be broken.** Four drivers retired in this race. The model had assigned every driver in the field a DNF probability of 0.0%-0.5% — meaningfully lower than reality for the entire grid, not just an unlucky miss. This points to class imbalance in training (DNFs are a minority outcome, so a plain logistic regression can learn to predict "unlikely" almost everywhere and still look correct on average) rather than a genuine per-driver risk signal. Full comparison in [`spanish_gp_result.md`](./spanish_gp_result.md).

## What I'd try next

- **Fix the DNF model** using class-weighting (`class_weight='balanced'`) or resampling, then re-validate against both live test races retroactively
- Incorporate practice session pace as an additional pre-qualifying signal
- Add weather data, since wet-race unpredictability likely explains some of the model's mid-pack weakness
- Try a non-linear model (e.g. random forest) to see if it captures driver/track interactions better than linear regression
- Keep tracking live predictions across a full season to build a statistically reliable accuracy picture, rather than judging from one or two test races

## Project structure

```
build_dataset.py         - Builds/resumes the historical dataset (2022-2025)
predict_position.py      - Baseline and recent-form position models
check_race.py            - Inspects model predictions against a specific past race
predict_future_race.py   - Predicts a race using only pre-race information
predict_next_race.py     - Live prediction: pulls real qualifying data automatically,
                            includes DNF risk model
f1_dataset.csv            - The built historical dataset
f1_predictions.csv        - Enriched dataset with predictions and accuracy labels
spanish_gp_result.md      - Predicted vs actual comparison for the live Spanish GP test
Italian grand prix 2026 predictions.png  - Screenshot of the Italian GP live prediction
Spanish grand prix predictions.png       - Screenshot of the Spanish GP live prediction
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
