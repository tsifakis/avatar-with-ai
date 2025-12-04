# Temperature Forecasting Script Evaluation

## Overview
The `temperature_forecasting.py` script builds a small stacked LSTM to predict short-term temperature changes using handcrafted forward/backward-difference features. The workflow seeds randomness, prepares sliding-window datasets, trains the network, reports regression metrics, forecasts the next value, and writes a comparison plot to `artifacts/pred_vs_actual.png`.

## Strengths
- **Reproducibility:** Seeds NumPy and TensorFlow, ensuring consistent data splits and initialization for easier debugging and comparison.
- **Feature engineering clarity:** Explicit forward and backward differences make the input construction transparent and match the original intent of capturing local trends.
- **Non-interactive pipeline:** Avoids blocking `plt.show()` calls by saving plots to disk, which is CI-friendly and easier to inspect later.
- **Modular functions:** Breaks out feature prep, model building, callbacks, forecasting, and plotting into testable units.

## Weaknesses and Risks
- **Tiny dataset:** With ~150 points and a 3-layer LSTM, the model is highly likely to overfit and yield unstable generalization metrics.
- **Data leakage potential:** The target uses the raw series value while the features use first differences; however, no scaling/normalization is applied, so value ranges and stationarity are not enforced.
- **Train/validation overlap risk:** The validation split is drawn from the training subset after a random split, which can still produce optimistic estimates on such a small dataset; time-aware splits would better reflect forecasting use cases.
- **Lack of persistence:** The trained model is not saved, so repeat runs retrain from scratch; no artifact is emitted beyond the plot.
- **Minimal evaluation:** Only a single random split is evaluated; no cross-validation or rolling-origin evaluation is performed, making results sensitive to the split seed.

## Recommendations
- Add scaling (e.g., `StandardScaler`) to both features and targets, fitting on the training portion only to avoid leakage.
- Use a temporal split (or rolling-origin evaluation) instead of random `train_test_split` to better mirror forecasting conditions.
- Save the trained model (e.g., `model.save('artifacts/model.keras')`) and consider logging metrics/plots in a structured way for reproducibility.
- Experiment with simpler baselines (persistence, linear regression, small MLP) to benchmark the LSTM and justify complexity.
- Reduce network depth/size or add regularization (e.g., L2, dropout tuning, early stopping) given the limited data volume.
