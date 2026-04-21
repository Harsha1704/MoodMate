import os
import json
import shutil
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .preprocess import (
    build_datasets_from_folders,
    get_class_weight_dict,
    EMOTION_LABELS,
    IMG_SIZE,
    NUM_CLASSES,
)

SEED = 42
BATCH_SIZE = 32

EPOCHS_STAGE1 = 20
EPOCHS_STAGE2 = 15

HEAD_LR = 3e-4
FT_LR = 1e-5

WEIGHT_DECAY_STAGE1 = 1e-4
WEIGHT_DECAY_STAGE2 = 5e-5

LABEL_SMOOTHING_STAGE1 = 0.05
LABEL_SMOOTHING_STAGE2 = 0.03

UNFREEZE_LAST_N = 30
USE_CLASS_WEIGHTS_STAGE2 = True
ALWAYS_TRAIN_FRESH = True
COPY_STAGE1_IF_STAGE2_WORSE = True

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "fer2013")
MODELS_DIR = os.path.join(BASE_DIR, "models")

MODEL_PATH = os.path.join(MODELS_DIR, "emotion_model.keras")
STAGE1_MODEL_PATH = os.path.join(MODELS_DIR, "emotion_model_stage1.keras")
STAGE2_MODEL_PATH = os.path.join(MODELS_DIR, "emotion_model_stage2.keras")

TRAIN_LOG_STAGE1 = os.path.join(MODELS_DIR, "training_log_stage1.csv")
TRAIN_LOG_STAGE2 = os.path.join(MODELS_DIR, "training_log_stage2.csv")

REPORT_TXT = os.path.join(MODELS_DIR, "classification_report.txt")
CM_PNG = os.path.join(MODELS_DIR, "confusion_matrix.png")
CURVES_PNG = os.path.join(MODELS_DIR, "training_curves.png")
SUMMARY_JSON = os.path.join(MODELS_DIR, "training_summary.json")

os.makedirs(MODELS_DIR, exist_ok=True)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


def clear_old_checkpoints():
    paths = [
        STAGE1_MODEL_PATH,
        STAGE2_MODEL_PATH,
        MODEL_PATH,
        TRAIN_LOG_STAGE1,
        TRAIN_LOG_STAGE2,
        REPORT_TXT,
        CM_PNG,
        CURVES_PNG,
        SUMMARY_JSON,
    ]
    for path in paths:
        if os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass


def build_augmentation():
    return keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.05),
            layers.RandomZoom(0.05),
            layers.RandomTranslation(0.05, 0.05),
        ],
        name="augmentation",
    )


def build_model():
    augmentation = build_augmentation()

    backbone = keras.applications.EfficientNetB0(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    backbone.trainable = False

    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = augmentation(inputs)
    x = keras.applications.efficientnet.preprocess_input(x)
    x = backbone(x, training=False)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.35)(x)
    x = layers.Dense(256, activation="swish")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.30)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="Emotion_EfficientNetB0_Stable")
    return model


def build_optimizer(lr, weight_decay):
    return keras.optimizers.AdamW(
        learning_rate=lr,
        weight_decay=weight_decay,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7,
        clipnorm=1.0,
    )


def compile_stage1(model):
    optimizer = build_optimizer(HEAD_LR, WEIGHT_DECAY_STAGE1)
    loss = keras.losses.CategoricalCrossentropy(
        label_smoothing=LABEL_SMOOTHING_STAGE1
    )

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=[
            "accuracy",
            keras.metrics.TopKCategoricalAccuracy(k=2, name="top2_acc"),
        ],
    )
    return model


def compile_stage2(model):
    optimizer = build_optimizer(FT_LR, WEIGHT_DECAY_STAGE2)
    loss = keras.losses.CategoricalCrossentropy(
        label_smoothing=LABEL_SMOOTHING_STAGE2
    )

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=[
            "accuracy",
            keras.metrics.TopKCategoricalAccuracy(k=2, name="top2_acc"),
        ],
    )
    return model


def make_callbacks(save_path, csv_path, patience):
    return [
        keras.callbacks.ModelCheckpoint(
            save_path,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=0.5,
            patience=2,
            min_lr=1e-7,
            verbose=1,
        ),
        keras.callbacks.CSVLogger(csv_path),
    ]


def plot_training_curves(histories):
    acc, val_acc, loss, val_loss = [], [], [], []

    for h in histories:
        if h is None:
            continue
        acc.extend(h.history.get("accuracy", []))
        val_acc.extend(h.history.get("val_accuracy", []))
        loss.extend(h.history.get("loss", []))
        val_loss.extend(h.history.get("val_loss", []))

    if not acc:
        return

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(acc, label="Train Accuracy")
    plt.plot(val_acc, label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(loss, label="Train Loss")
    plt.plot(val_loss, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig(CURVES_PNG, dpi=200, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(cm):
    plt.figure(figsize=(8, 7))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar()

    ticks = np.arange(len(EMOTION_LABELS))
    plt.xticks(ticks, EMOTION_LABELS, rotation=45)
    plt.yticks(ticks, EMOTION_LABELS)

    thresh = cm.max() / 2.0 if cm.size else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(CM_PNG, dpi=200, bbox_inches="tight")
    plt.close()


def evaluate_model(model, ds):
    y_true, y_prob = [], []

    for batch_x, batch_y in ds:
        probs = model.predict(batch_x, verbose=0)
        y_prob.append(probs)
        y_true.append(np.argmax(batch_y.numpy(), axis=1))

    y_true = np.concatenate(y_true)
    y_prob = np.concatenate(y_prob)
    y_pred = np.argmax(y_prob, axis=1)

    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(
        y_true,
        y_pred,
        target_names=EMOTION_LABELS,
        digits=4,
        zero_division=0,
    )
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))

    loss, acc, top2 = model.evaluate(ds, verbose=0)
    return {
        "loss": float(loss),
        "accuracy": float(acc),
        "top2_acc": float(top2),
        "macro_f1": macro_f1,
        "cm": cm,
        "report": report,
    }


def save_evaluation_artifacts(metrics_dict):
    plot_confusion_matrix(metrics_dict["cm"])

    with open(REPORT_TXT, "w", encoding="utf-8") as f:
        f.write(metrics_dict["report"])
        f.write(f"\nMacro F1: {metrics_dict['macro_f1']:.4f}\n")
        f.write(f"Accuracy: {metrics_dict['accuracy']:.4f}\n")
        f.write(f"Loss: {metrics_dict['loss']:.4f}\n")
        f.write(f"Top-2 Accuracy: {metrics_dict['top2_acc']:.4f}\n")


def find_backbone_model(model):
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            name = layer.name.lower()
            if "efficientnet" in name:
                return layer

    raise ValueError(
        f"Could not find EfficientNet backbone. Top-level layers: {[layer.name for layer in model.layers]}"
    )


def set_backbone_trainable_tail(backbone, unfreeze_last_n):
    backbone.trainable = True
    all_layers = list(backbone.layers)
    split_idx = max(len(all_layers) - unfreeze_last_n, 0)

    for i, layer in enumerate(all_layers):
        layer.trainable = i >= split_idx
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False


def maybe_copy_best_stage1(stage1_path, stage2_path, stage1_metrics, stage2_metrics):
    if not COPY_STAGE1_IF_STAGE2_WORSE:
        return stage2_path

    if stage2_metrics["accuracy"] >= stage1_metrics["accuracy"]:
        return stage2_path

    shutil.copy2(stage1_path, MODEL_PATH)
    return stage1_path


def train_stage1(train_ds, val_ds):
    print("Training Stage 1 from scratch...")
    model = build_model()
    compile_stage1(model)

    callbacks = make_callbacks(
        save_path=STAGE1_MODEL_PATH,
        csv_path=TRAIN_LOG_STAGE1,
        patience=5,
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_STAGE1,
        callbacks=callbacks,
        verbose=1,
    )

    best_model = keras.models.load_model(STAGE1_MODEL_PATH, compile=False)
    compile_stage1(best_model)
    return best_model, history


def fine_tune_stage2(model, train_ds, val_ds, class_weights):
    backbone = find_backbone_model(model)
    print(f"Using backbone: {backbone.name}")
    set_backbone_trainable_tail(backbone, UNFREEZE_LAST_N)

    compile_stage2(model)

    callbacks = make_callbacks(
        save_path=STAGE2_MODEL_PATH,
        csv_path=TRAIN_LOG_STAGE2,
        patience=4,
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_STAGE2,
        class_weight=class_weights if USE_CLASS_WEIGHTS_STAGE2 else None,
        callbacks=callbacks,
        verbose=1,
    )

    best_model = keras.models.load_model(STAGE2_MODEL_PATH, compile=False)
    compile_stage2(best_model)
    return best_model, history


def train():
    if ALWAYS_TRAIN_FRESH:
        clear_old_checkpoints()

    train_ds, val_ds, train_counts, _ = build_datasets_from_folders(
        DATA_DIR,
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        seed=SEED,
    )

    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    class_weights = get_class_weight_dict(train_counts)
    histories = []

    model_stage1, history_stage1 = train_stage1(train_ds, val_ds)
    histories.append(history_stage1)

    stage1_metrics = evaluate_model(model_stage1, val_ds)
    print(f"Stage 1 accuracy: {stage1_metrics['accuracy'] * 100:.2f}%")
    print(f"Stage 1 macro F1: {stage1_metrics['macro_f1']:.4f}")

    model_stage2, history_stage2 = fine_tune_stage2(
        model=model_stage1,
        train_ds=train_ds,
        val_ds=val_ds,
        class_weights=class_weights,
    )
    histories.append(history_stage2)

    stage2_metrics = evaluate_model(model_stage2, val_ds)
    print(f"Stage 2 accuracy: {stage2_metrics['accuracy'] * 100:.2f}%")
    print(f"Stage 2 macro F1: {stage2_metrics['macro_f1']:.4f}")

    final_source_path = maybe_copy_best_stage1(
        stage1_path=STAGE1_MODEL_PATH,
        stage2_path=STAGE2_MODEL_PATH,
        stage1_metrics=stage1_metrics,
        stage2_metrics=stage2_metrics,
    )

    final_model = keras.models.load_model(final_source_path, compile=False)

    if final_source_path == STAGE1_MODEL_PATH:
        compile_stage1(final_model)
        best_stage = "stage1"
    else:
        compile_stage2(final_model)
        best_stage = "stage2"

    final_metrics = evaluate_model(final_model, val_ds)
    save_evaluation_artifacts(final_metrics)
    final_model.save(MODEL_PATH)
    plot_training_curves(histories)

    summary = {
        "saved_model_path": MODEL_PATH,
        "best_stage": best_stage,
        "stage1_accuracy": stage1_metrics["accuracy"],
        "stage1_macro_f1": stage1_metrics["macro_f1"],
        "stage2_accuracy": stage2_metrics["accuracy"],
        "stage2_macro_f1": stage2_metrics["macro_f1"],
        "final_accuracy": final_metrics["accuracy"],
        "final_loss": final_metrics["loss"],
        "final_macro_f1": final_metrics["macro_f1"],
        "final_top2_acc": final_metrics["top2_acc"],
        "img_size": IMG_SIZE,
        "batch_size": BATCH_SIZE,
        "epochs_stage1": EPOCHS_STAGE1,
        "epochs_stage2": EPOCHS_STAGE2,
        "head_lr": HEAD_LR,
        "ft_lr": FT_LR,
        "unfreeze_last_n": UNFREEZE_LAST_N,
        "use_class_weights_stage2": USE_CLASS_WEIGHTS_STAGE2,
    }

    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nFinal selected accuracy: {final_metrics['accuracy'] * 100:.2f}%")
    print(f"Final selected loss: {final_metrics['loss']:.4f}")
    print(f"Final selected macro F1: {final_metrics['macro_f1']:.4f}")
    print(f"Best stage selected: {best_stage}")
    print(f"Saved final model to: {MODEL_PATH}")
    print(f"Saved report to: {REPORT_TXT}")
    print(f"Saved confusion matrix to: {CM_PNG}")
    print(f"Saved training curves to: {CURVES_PNG}")


if __name__ == "__main__":
    train()