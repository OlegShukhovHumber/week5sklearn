# Late Deliveries

QuickBite wants to know, when an order is placed, whether it will arrive more than ten minutes late, so it can offer a $4 voucher first. This repository builds a scikit-learn pipeline that predicts the chance an order is late and flags it when that chance passes a threshold chosen from the company's own costs.

## Setup

```
conda env create -f environment.yml
conda activate lab05-late-deliveries
pip install -r requirements.txt
pip install -e .
```

## Run

```
python -m latedeliveries.train
```

This recreates `reports/cv_results.csv`, `reports/metrics.json`, `models/late_orders.joblib` and `models/model_info.json` (plus `reports/tuning.json`, `reports/threshold_costs.csv` and `reports/leakage_check.json`, which the notebook reads). The data come from `make_orders()` with its default seed.

## Check it

```
python -m ruff format --check .
python -m ruff check .
python -m pytest
```

## Decisions

**Columns used.** `distance_km`, `items`, `hour`, `prep_minutes`, `courier_months` (numeric) and `weather`, `day_type`, `vehicle` (categorical). The feature lists are defined once, in `pipeline.py`.

**Columns left out.**
- `customer_rating`: the customer gives it after the delivery, so it is not known when the order is placed, and it is largely a consequence of the order being late. Leakage check (training data, five-fold CV, same model): ROC-AUC is 0.795 without it and 0.907 with it, a jump of 0.112. A jump that large from one column is a warning, so it stays out.
- `order_id`: only a label with no meaning.
- `late`: the target.

`prep_minutes` is a judgement call. I kept it, treating it as the kitchen's estimate at the time of ordering; the brief also expects the pipeline to cope with it being missing. If it is only known once the food is cooked, it would be too late to use and this model would look slightly better than it really is. `courier_months` and `vehicle` carry a similar assumption: the courier is known when the order is placed.

**Model choice.** Five-fold stratified cross-validation on the training data only, every model inside the same pipeline. Mean ± standard deviation across folds; precision, recall and F1 use the default 0.5 cut-off. The last two columns apply each model's own best threshold to out-of-fold predictions.

| model | accuracy | precision | recall | F1 | ROC-AUC | best threshold | cost ($) |
|---|---|---|---|---|---|---|---|
| dummy (baseline) | 0.844 ± 0.001 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.500 ± 0.000 | 0.20 | 5265 |
| logistic regression | 0.862 ± 0.009 | 0.654 ± 0.065 | 0.242 ± 0.053 | 0.352 ± 0.064 | 0.795 ± 0.034 | 0.30 | **4031** |
| logistic regression, balanced | 0.737 ± 0.019 | 0.333 ± 0.027 | 0.681 ± 0.049 | 0.447 ± 0.034 | 0.795 ± 0.034 | 0.70 | 4098 |
| random forest | 0.860 ± 0.005 | 0.754 ± 0.064 | 0.151 ± 0.032 | 0.250 ± 0.046 | 0.786 ± 0.036 | 0.25 | 4180 |
| gradient boosting | 0.851 ± 0.006 | 0.544 ± 0.036 | 0.288 ± 0.037 | 0.375 ± 0.036 | 0.754 ± 0.024 | 0.30 | 4327 |

Accuracy is no help: never flagging anything already scores 84%. I chose by cost, because a wasted voucher ($4) and a missed late order ($15) are not equally bad. Plain logistic regression has the lowest cost and is the simplest model. The balanced version has the same ROC-AUC to three decimals (class weights mostly shift the scores, not the ranking) and its best threshold moves to 0.70, so balancing only shifts where the cut-off sits. The gaps between models in cost are small next to the fold-to-fold spread in ROC-AUC (about 0.03), so I would not claim logistic regression is clearly better than the balanced version or the random forest, only that nothing beat it and it is the simplest.

**What tuning changed.** GridSearchCV over `model__C` (0.01, 0.1, 1, 10) and `model__class_weight` (none, balanced), eight combinations, scored by ROC-AUC because it judges the ranking of orders before any threshold is chosen. Best: `C=10`, no class weight. ROC-AUC went from 0.7951 to 0.7952, a gain of 0.0002 against a fold-to-fold spread of 0.034. Tuning did not beat the untuned model by more than the spread, so it changed nothing that matters.

**Threshold.** Out-of-fold probabilities on the training data (`cross_val_predict`), thresholds 0.05 to 0.95 in steps of 0.05, lowest total cost. Result: 0.30 (training cost $4035; 0.25 costs $4087 and 0.35 costs $4113). The cost rule needs its own threshold because the default 0.5 ignores that the two mistakes cost different amounts. A voucher costs $4 and a missed late order $15, so flagging is worth it once the chance of lateness passes 4/15, about 27%, which is where the chosen value sits. The test set had no say in it.

## Result

Test set, 750 orders, 117 of them late (15.6%), model fitted on the whole training set and scored once at threshold 0.30:

| | predicted on time | flagged |
|---|---|---|
| actually on time | 578 | 55 |
| actually late | 64 | 53 |

Precision 0.491, recall 0.453, F1 0.471, ROC-AUC 0.796 (accuracy 0.841).

| policy | total cost |
|---|---|
| flag nobody | $1,755 |
| flag everybody | $3,000 |
| model, threshold 0.30 | $1,392 |

The model saves $363 (21%) against flagging nobody. I would trust the direction (the test ROC-AUC matches cross-validation) but not the exact figure: only 117 late orders decide it, and ROC-AUC varies by about 0.03 between folds.

## What the tests cover

16 tests. The feature list excludes the leaky columns; the split is stratified and repeatable; the pipeline gives finite probabilities for a missing `prep_minutes` and a weather value it has never seen; predicted probabilities are valid for two models (parametrized); the cost function, the threshold choice, the three policy costs and the confusion matrix match a six-order example worked out on paper, including the two extremes (flag nobody, flag everybody); a saved pipeline predicts identically after reloading. I did not test `train.py` end to end: it takes several seconds and the same numbers are already committed in `reports/`, which is checked by running it on a fresh clone.

## Known limitations

- `prep_minutes` and the courier columns may not all be known at order time (see Decisions).
- The model finds under half of the late orders and about half of its flags are vouchers that were not needed.
- Costs are fixed at $4 and $15 and the data are invented; a real rollout would need to check both.
- One split, one seed. With more time I would repeat the split with several seeds to see how stable the saving is, and calibrate the probabilities before trusting the 27% rule of thumb.
