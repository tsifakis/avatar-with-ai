"""Train an LSTM model to forecast short-term temperature changes.

The script replicates the dataset and modeling choices from the provided
notebook-style snippet while making the workflow reproducible and
non-interactive (plots are saved instead of displayed).
"""
from __future__ import annotations

from math import sqrt
from pathlib import Path
from typing import Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from colorama import Fore, Style
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential

SEED = 1
tf.random.set_seed(SEED)
np.random.seed(SEED)


# Dataset of normalized temperatures.
TEMPERATURE_SERIES = np.array(
    [
        0.48,
        0.43,
        0.5,
        0.5,
        0.33,
        0.31,
        0.33,
        0.52,
        0.45,
        0.53,
        0.43,
        0.43,
        0.46,
        0.45,
        0.31,
        0.43,
        0.49,
        0.38,
        0.48,
        0.42,
        0.4,
        0.43,
        0.5,
        0.45,
        0.26,
        0.52,
        0.36,
        0.35,
        0.4,
        0.43,
        0.59,
        0.39,
        0.28,
        0.41,
        0.36,
        0.46,
        0.46,
        0.29,
        0.47,
        0.37,
        0.46,
        0.47,
        0.54,
        0.5,
        0.44,
        0.28,
        0.46,
        0.43,
        0.36,
        0.38,
        0.39,
        0.38,
        0.37,
        0.48,
        0.42,
        0.33,
        0.33,
        0.46,
        0.4,
        0.59,
        0.33,
        0.41,
        0.41,
        0.45,
        0.45,
        0.36,
        0.38,
        0.39,
        0.52,
        0.4,
        0.39,
        0.45,
        0.44,
        0.56,
        0.34,
        0.35,
        0.38,
        0.6,
        0.53,
        0.44,
        0.39,
        0.43,
        0.46,
        0.5,
        0.33,
        0.3,
        0.55,
        0.45,
        0.36,
        0.32,
        0.41,
        0.49,
        0.35,
        0.42,
        0.34,
        0.51,
        0.41,
        0.43,
        0.43,
        0.41,
        0.55,
        0.5,
        0.45,
        0.42,
        0.59,
        0.38,
        0.51,
        0.28,
        0.52,
        0.49,
        0.47,
        0.45,
        0.41,
        0.41,
        0.46,
        0.51,
        0.42,
        0.52,
        0.54,
        0.48,
        0.45,
        0.44,
        0.49,
        0.35,
        0.48,
        0.48,
        0.39,
        0.25,
        0.36,
        0.37,
        0.44,
        0.39,
        0.45,
        0.54,
        0.28,
        0.3,
        0.3,
        0.43,
        0.41,
        0.31,
        0.43,
        0.33,
        0.44,
        0.46,
        0.41,
        0.42,
        0.47,
        0.5,
        0.4,
        0.44,
        0.35,
        0.34,
        0.43,
        0.47,
        0.44,
        0.34,
        0.34,
        0.48,
        0.37,
        0.5,
        0.33,
        0.41,
        0.53,
    ],
    dtype=np.float32,
)


def calculate_forward_differences(data: np.ndarray) -> np.ndarray:
    """Return forward differences with the last difference repeated."""
    return np.diff(data, append=data[-1])


def calculate_backward_differences(data: np.ndarray) -> np.ndarray:
    """Return backward differences with the first difference repeated."""
    return np.diff(data, prepend=data[0])


def create_dataset(data: np.ndarray, steps: int, output: int) -> Tuple[np.ndarray, np.ndarray]:
    """Build sliding window datasets for sequence-to-one prediction."""
    X, y = [], []
    for i in range(len(data) - steps - output + 1):
        seq_x = data[i : i + steps]
        seq_y = data[i + steps : i + steps + output, 0]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


def build_model(steps: int, features: int, output: int) -> Sequential:
    """Construct the stacked LSTM model used for training."""
    model = Sequential(
        [
            LSTM(
                1,
                activation="tanh",
                return_sequences=True,
                recurrent_activation="sigmoid",
                input_shape=(steps, features),
                recurrent_dropout=0.1,
                bias_initializer="zeros",
                kernel_initializer="glorot_uniform",
                recurrent_initializer="orthogonal",
                go_backwards=False,
                stateful=False,
                implementation=2,
                unroll=False,
                unit_forget_bias=True,
            ),
            Dropout(0.1),
            LSTM(
                2,
                activation="tanh",
                return_sequences=True,
                recurrent_activation="sigmoid",
                recurrent_dropout=0.1,
                bias_initializer="zeros",
                kernel_initializer="glorot_uniform",
                recurrent_initializer="orthogonal",
                go_backwards=False,
                stateful=False,
                implementation=2,
                unroll=False,
                unit_forget_bias=True,
            ),
            Dropout(0.1),
            LSTM(
                1,
                activation="tanh",
                return_sequences=False,
                recurrent_activation="sigmoid",
                recurrent_dropout=0.1,
                bias_initializer="zeros",
                kernel_initializer="glorot_uniform",
                recurrent_initializer="orthogonal",
                go_backwards=False,
                stateful=False,
                implementation=2,
                unroll=False,
                unit_forget_bias=True,
            ),
            Dropout(0.1),
            Dense(
                output,
                activation="tanh",
                use_bias=True,
                bias_initializer="zeros",
                kernel_initializer="glorot_uniform",
            ),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[tf.keras.metrics.MeanSquaredError()],
    )
    return model


def create_callbacks() -> Iterable[ReduceLROnPlateau]:
    return [
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.1,
            patience=15,
            verbose=1,
            mode="min",
            min_delta=0.00001,
            cooldown=0,
            min_lr=0,
        )
    ]


def prepare_features(series: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int, int]:
    forward = calculate_forward_differences(series)
    backward = calculate_backward_differences(series)
    diffs = np.column_stack((series[:-1], forward[:-1], backward[:-1]))
    steps = 1
    output = 1
    features = diffs.shape[1]
    X, y = create_dataset(diffs, steps, output)
    X = X.reshape((X.shape[0], steps, features))
    return X, y, steps, output


def forecast_next(series: np.ndarray, model: Sequential) -> float:
    forward = calculate_forward_differences(series)[-1]
    backward = calculate_backward_differences(series)[-1]
    last_steps = np.array([[series[-1], forward, backward]], dtype=np.float32).reshape((1, 1, 3))
    prediction = model.predict(last_steps, verbose=0)
    return float(prediction[0][0])


def save_comparison_plot(y_true: np.ndarray, y_pred: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(y_true.flatten(), label="Πραγματικές Τιμές", color="blue")
    plt.plot(y_pred.flatten(), label="Προβλεπόμενες Τιμές", color="red", alpha=0.6)
    plt.title("Σύγκριση Πραγματικών και Προβλεπόμενων Τιμών")
    plt.xlabel("Δείγμα")
    plt.ylabel("Τιμή")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)


def train_and_evaluate(series: np.ndarray) -> None:
    X, y, steps, output = prepare_features(series)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED
    )
    model = build_model(steps, X.shape[2], output)
    model.summary()

    model.fit(
        X_train,
        y_train,
        epochs=50,
        batch_size=1,
        verbose=2,
        validation_split=0.2,
        callbacks=list(create_callbacks()),
    )

    y_pred = model.predict(X_test, verbose=0)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"\n{Fore.YELLOW}Mean   Absolute  Error   (MAE): {Fore.BLUE}{mae}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}Mean   Squared   Error   (MSE): {Fore.BLUE}{mse}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}Root Mean Squared Error (RMSE): {Fore.BLUE}{rmse}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}R^2   Score        (r2_score) : {Fore.BLUE}{r2}{Style.RESET_ALL}")

    next_temp = forecast_next(series, model) * 100
    final_prediction = int(round(next_temp))
    print(f"\nΟΙ ΕΠΟΜΕΝΕΣ ΘΕΡΜΟΚΡΑΣΙΕΣ: {final_prediction}")

    plot_path = Path("artifacts") / "pred_vs_actual.png"
    save_comparison_plot(y_test, y_pred, plot_path)
    print(f"Saved comparison plot to {plot_path.resolve()}")


if __name__ == "__main__":
    train_and_evaluate(TEMPERATURE_SERIES)
