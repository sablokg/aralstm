#!/usr/bin/env python3

import pickle

import click
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout, Embedding
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer


def seq_to_kmers(seq, k=6):
    seq = seq.upper()
    return [seq[i : i + k] for i in range(len(seq) - k + 1)]


def encode_sequences(sequences, tokenizer, k, maxlen=None):
    docs = [" ".join(seq_to_kmers(s, k)) for s in sequences]
    encoded = tokenizer.texts_to_sequences(docs)
    return pad_sequences(encoded, maxlen=maxlen, padding="post")


def aralstm(
    pathfile: str,
    k: int = 6,
    embedding_dim: int = 64,
    lstm_units: int = 32,
    dropout: float = 0.3,
    test_size: float = 0.25,
    epochs: int = 100,
    batch_size: int = 4,
    patience: int = 8,
    random_state: int = 42,
    model_out: str = "dna_kmer_lstm_final.keras",
    checkpoint_out: str = "dna_kmer_lstm_best.keras",
    tokenizer_out: str = "tokenizer.pkl",
    meta_out: str = "meta.pkl",
    training_curves_out: str = "training_curves.png",
    confusion_matrix_out: str = "confusion_matrix.png",
):
    sequences = []
    labels = []
    with open(pathfile, "r") as f:
        for line in f:
            if line.startswith(">"):
                labels.append(line.strip())
            else:
                sequences.append(line.strip())

    num_classes = len(set(labels))
    K = k

    # -----------------------------
    # 2. Fit tokenizer on ALL sequences (train+val), then split
    # -----------------------------
    kmer_docs = [" ".join(seq_to_kmers(s, K)) for s in sequences]
    tokenizer = Tokenizer(lower=False, oov_token="<OOV>")
    tokenizer.fit_on_texts(kmer_docs)
    vocab_size = len(tokenizer.word_index) + 1
    X_all = pad_sequences(tokenizer.texts_to_sequences(kmer_docs), padding="post")
    maxlen = X_all.shape[1]
    y_all = np.array(labels)
    X_train, X_val, y_train, y_val = train_test_split(
        X_all, y_all, test_size=test_size, stratify=y_all, random_state=random_state
    )

    # -----------------------------
    # 3. Build & train model
    # -----------------------------
    model = Sequential(
        [
            Embedding(
                input_dim=vocab_size, output_dim=embedding_dim, input_length=maxlen
            ),
            Bidirectional(LSTM(lstm_units)),
            Dropout(dropout),
            Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        EarlyStopping(
            monitor="val_loss", patience=patience, restore_best_weights=True, verbose=1
        ),
        ModelCheckpoint(
            filepath=checkpoint_out,
            monitor="val_loss",
            save_best_only=True,
            verbose=0,
        ),
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    # -----------------------------
    # 4. Plot training curves
    # -----------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["loss"], label="train loss")
    axes[0].plot(history.history["val_loss"], label="val loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss over epochs")
    axes[0].legend()

    axes[1].plot(history.history["accuracy"], label="train acc")
    axes[1].plot(history.history["val_accuracy"], label="val acc")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy over epochs")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(training_curves_out, dpi=150)
    plt.close()

    # -----------------------------
    # 5. Confusion matrix on validation set
    # -----------------------------
    val_preds = model.predict(X_val)
    val_pred_classes = np.argmax(val_preds, axis=1)

    cm = confusion_matrix(y_val, val_pred_classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues")
    plt.title("Validation Confusion Matrix")
    plt.savefig(confusion_matrix_out, dpi=150)
    plt.close()

    click.echo("\nClassification report (validation set):")
    click.echo(classification_report(y_val, val_pred_classes, zero_division=0))

    # -----------------------------
    # 6. Save model, tokenizer, metadata
    # -----------------------------
    model.save(model_out)

    with open(tokenizer_out, "wb") as f:
        pickle.dump(tokenizer, f)

    meta = {"k": K, "maxlen": maxlen, "num_classes": num_classes}
    with open(meta_out, "wb") as f:
        pickle.dump(meta, f)

    click.echo(f"\nSaved: {model_out}, {tokenizer_out}, {meta_out}")

    return model_out, tokenizer_out, meta_out


def run_predictions(model_path, tokenizer_path, meta_path, seqs):
    loaded_model = load_model(model_path)
    with open(tokenizer_path, "rb") as f:
        loaded_tokenizer = pickle.load(f)
    with open(meta_path, "rb") as f:
        loaded_meta = pickle.load(f)

    X_new = encode_sequences(
        seqs, loaded_tokenizer, loaded_meta["k"], maxlen=loaded_meta["maxlen"]
    )

    preds = loaded_model.predict(X_new)
    pred_classes = np.argmax(preds, axis=1)
    pred_conf = np.max(preds, axis=1)

    click.echo("\nPredictions:")
    for seq, cls, conf in zip(seqs, pred_classes, pred_conf):
        click.echo(f"  {seq[:20]}...  -> class {cls}  (confidence {conf:.3f})")

    return list(zip(seqs, pred_classes, pred_conf))


# =====================================================================
# CLI
# =====================================================================
@click.group()
def cli():
    """DNA k-mer BiLSTM classifier: train a model and predict on new sequences."""
    pass


@cli.command()
@click.argument("pathfile", type=click.Path(exists=True))
@click.option("-k", "--kmer-size", default=6, show_default=True, help="k-mer length.")
@click.option(
    "--embedding-dim", default=64, show_default=True, help="Embedding output dimension."
)
@click.option(
    "--lstm-units", default=32, show_default=True, help="Units per LSTM direction."
)
@click.option("--dropout", default=0.3, show_default=True, help="Dropout rate.")
@click.option(
    "--test-size", default=0.25, show_default=True, help="Validation split fraction."
)
@click.option("--epochs", default=100, show_default=True, help="Max training epochs.")
@click.option("--batch-size", default=4, show_default=True, help="Training batch size.")
@click.option(
    "--patience", default=8, show_default=True, help="EarlyStopping patience."
)
@click.option(
    "--random-state",
    default=42,
    show_default=True,
    help="Random seed for train/val split.",
)
@click.option(
    "--model-out",
    default="dna_kmer_lstm_final.keras",
    show_default=True,
    help="Path to save final model.",
)
@click.option(
    "--checkpoint-out",
    default="dna_kmer_lstm_best.keras",
    show_default=True,
    help="Path to save best checkpoint.",
)
@click.option(
    "--tokenizer-out",
    default="tokenizer.pkl",
    show_default=True,
    help="Path to save tokenizer.",
)
@click.option(
    "--meta-out", default="meta.pkl", show_default=True, help="Path to save metadata."
)
@click.option(
    "--curves-out",
    "training_curves_out",
    default="training_curves.png",
    show_default=True,
    help="Path to save training curve plot.",
)
@click.option(
    "--cm-out",
    "confusion_matrix_out",
    default="confusion_matrix.png",
    show_default=True,
    help="Path to save confusion matrix plot.",
)
@click.option(
    "--predict-seq",
    "predict_seqs",
    multiple=True,
    help="Optional sequence(s) to predict on after training. Repeatable.",
)
def train(
    pathfile,
    kmer_size,
    embedding_dim,
    lstm_units,
    dropout,
    test_size,
    epochs,
    batch_size,
    patience,
    random_state,
    model_out,
    checkpoint_out,
    tokenizer_out,
    meta_out,
    training_curves_out,
    confusion_matrix_out,
    predict_seqs,
):
    """Train the BiLSTM classifier on sequences in PATHFILE (FASTA-like: '>' header lines are labels)."""
    model_out, tokenizer_out, meta_out = aralstm(
        pathfile,
        k=kmer_size,
        embedding_dim=embedding_dim,
        lstm_units=lstm_units,
        dropout=dropout,
        test_size=test_size,
        epochs=epochs,
        batch_size=batch_size,
        patience=patience,
        random_state=random_state,
        model_out=model_out,
        checkpoint_out=checkpoint_out,
        tokenizer_out=tokenizer_out,
        meta_out=meta_out,
        training_curves_out=training_curves_out,
        confusion_matrix_out=confusion_matrix_out,
    )

    seqs = list(predict_seqs) or [
        "ACGTACGTTGCAACGTTGCATTAAA",
        "GGATCCAAGGCTGGATCCTTGGAAA",
    ]
    run_predictions(model_out, tokenizer_out, meta_out, seqs)


@cli.command()
@click.option(
    "--model",
    "model_path",
    default="dna_kmer_lstm_final.keras",
    show_default=True,
    type=click.Path(exists=True),
    help="Path to trained model.",
)
@click.option(
    "--tokenizer",
    "tokenizer_path",
    default="tokenizer.pkl",
    show_default=True,
    type=click.Path(exists=True),
    help="Path to saved tokenizer.",
)
@click.option(
    "--meta",
    "meta_path",
    default="meta.pkl",
    show_default=True,
    type=click.Path(exists=True),
    help="Path to saved metadata.",
)
@click.option(
    "--seq", "seqs", multiple=True, help="Sequence to predict on. Repeatable."
)
@click.option(
    "--seq-file",
    type=click.Path(exists=True),
    default=None,
    help="File with one sequence per line (used if no --seq given).",
)
def predict(model_path, tokenizer_path, meta_path, seqs, seq_file):
    """Predict classes for new sequences using a previously trained model."""
    seqs = list(seqs)
    if not seqs and seq_file:
        with open(seq_file, "r") as f:
            seqs = [line.strip() for line in f if line.strip()]
    if not seqs:
        raise click.UsageError("Provide at least one sequence via --seq or --seq-file.")

    run_predictions(model_path, tokenizer_path, meta_path, seqs)


if __name__ == "__main__":
    cli()
