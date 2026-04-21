import os
import sys
import json
import gc
import optuna
import tensorflow as tf
from tensorflow import keras

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.train_model import (
    load_data,
    build_model,
    compile_model,
    make_datasets,
    get_class_weights,
    BEST_PARAMS_PATH,
    MODELS_DIR,
)

TUNING_EPOCHS = 4
N_TRIALS = 3


def objective(trial):
    params = {
        "base_filters": trial.suggest_categorical("base_filters", [32, 48, 64]),
        "dense_units": trial.suggest_categorical("dense_units", [192, 256, 384]),
        "dropout1": trial.suggest_float("dropout1", 0.15, 0.30),
        "dropout2": trial.suggest_float("dropout2", 0.20, 0.35),
        "dropout3": trial.suggest_float("dropout3", 0.25, 0.40),
        "dropout4": trial.suggest_float("dropout4", 0.30, 0.50),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 8e-4, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [64, 96]),
        "rotation": trial.suggest_float("rotation", 0.04, 0.12),
        "zoom": trial.suggest_float("zoom", 0.04, 0.12),
        "translate": trial.suggest_float("translate", 0.04, 0.12),
        "contrast": trial.suggest_float("contrast", 0.03, 0.12),
    }

    X_train, y_train, y_train_cat, X_test, y_test, y_test_cat = load_data()
    class_weights = get_class_weights(y_train)

    model = build_model(
        base_filters=params["base_filters"],
        dense_units=params["dense_units"],
        dropout1=params["dropout1"],
        dropout2=params["dropout2"],
        dropout3=params["dropout3"],
        dropout4=params["dropout4"],
    )
    model = compile_model(model, learning_rate=params["learning_rate"])

    train_ds, val_ds = make_datasets(
        X_train,
        y_train_cat,
        X_test,
        y_test_cat,
        batch_size=params["batch_size"],
        rotation=params["rotation"],
        zoom=params["zoom"],
        translate=params["translate"],
        contrast=params["contrast"],
    )

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=4,
            restore_best_weights=True,
            verbose=0,
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=TUNING_EPOCHS,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=0,
    )

    best_val_acc = max(history.history["val_accuracy"])

    tf.keras.backend.clear_session()
    gc.collect()

    return best_val_acc


def run_optuna():
    os.makedirs(MODELS_DIR, exist_ok=True)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=N_TRIALS)

    print("\nBest validation accuracy:", study.best_value)
    print("Best parameters:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    with open(BEST_PARAMS_PATH, "w", encoding="utf-8") as f:
        json.dump(study.best_params, f, indent=2)

    print(f"\nSaved best params to: {BEST_PARAMS_PATH}")


if __name__ == "__main__":
    run_optuna()