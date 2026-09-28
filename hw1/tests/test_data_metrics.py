import numpy as np
import pandas as pd
import pytest

from hw1 import metrics
from hw1.data import DATA_ROOT, LABELS, dataset_dir, load_audio, load_manifest, load_split


@pytest.mark.parametrize("key", ["A", "B"])
def test_load_manifest_columns_and_y(key):
    df = load_manifest(key)
    for col in ["sample_id", "split", "label", "audio_path", "duration_seconds", "sample_rate", "sha256", "path", "y"]:
        assert col in df.columns
    assert set(df["split"]) <= {"train", "validation", "test"}
    test_rows = df[df["split"] == "test"]
    assert (test_rows["label"] == "").all()
    assert (test_rows["y"] == -1).all()
    labeled = df[df["split"] != "test"]
    assert set(labeled["label"]) <= set(LABELS[key])
    assert labeled["y"].min() >= 0
    assert labeled["y"].max() < len(LABELS[key])


def test_load_manifest_infers_key_from_dir():
    df_by_key = load_manifest("A")
    df_by_path = load_manifest(dataset_dir("A"))
    assert df_by_key["sample_id"].tolist() == df_by_path["sample_id"].tolist()


def test_load_split_filters_and_resets_index():
    df = load_split("A", "train")
    assert (df["split"] == "train").all()
    assert list(df.index) == list(range(len(df)))


@pytest.mark.parametrize("key", ["A", "B"])
def test_load_audio_shape_and_dtype(key):
    df = load_manifest(key)
    audio = load_audio(df.loc[0, "path"])
    assert audio.shape == (720000,)
    assert audio.dtype == np.float32


def test_topk_accuracy_simple_case():
    probs = np.array([[0.1, 0.7, 0.2], [0.5, 0.3, 0.2], [0.2, 0.2, 0.6]])
    y = np.array([1, 0, 0])
    assert metrics.topk_accuracy(probs, y, 1) == pytest.approx(2 / 3)
    assert metrics.topk_accuracy(probs, y, 3) == 1.0


def test_score_s_matches_formula():
    rng = np.random.default_rng(0)
    probs = rng.random((20, 6))
    y = rng.integers(0, 6, size=20)
    top1 = metrics.topk_accuracy(probs, y, 1)
    top3 = metrics.topk_accuracy(probs, y, 3)
    assert metrics.score_s(probs, y) == pytest.approx(top1 + 0.5 * top3)


def test_confusion_counts():
    y_true = np.array([0, 0, 1, 2])
    y_pred = np.array([0, 1, 1, 2])
    cm = metrics.confusion(y_true, y_pred, n_classes=3)
    expected = np.array([[1, 1, 0], [0, 1, 0], [0, 0, 1]])
    assert np.array_equal(cm, expected)


def test_evaluate_keys_and_shapes():
    rng = np.random.default_rng(1)
    probs = rng.random((10, 6))
    y = rng.integers(0, 6, size=10)
    result = metrics.evaluate(probs, y, LABELS["A"])
    assert set(result) == {"top1", "top3", "S", "cm"}
    assert len(result["cm"]) == 6
    assert all(len(row) == 6 for row in result["cm"])


def test_top3_labels_order():
    probs = np.array([[0.1, 0.5, 0.4, 0.0, 0.0, 0.0]])
    labels = LABELS["A"]
    result = metrics.top3_labels(probs, labels)
    assert result == [[labels[1], labels[2], labels[0]]]


def test_plot_confusion_writes_file(tmp_path):
    cm = np.array([[3, 1], [0, 4]])
    out = tmp_path / "confusion.png"
    metrics.plot_confusion(cm, ["a", "b"], out, normalize=True, title="test")
    assert out.exists()
    assert out.stat().st_size > 0


def test_write_and_validate_predictions(tmp_path):
    df_a = load_split("A", "test")
    ids = df_a["sample_id"].tolist()[:3]
    preds = {"dataset_A": {sid: LABELS["A"][:3] for sid in ids}}
    out = tmp_path / "preds.json"
    metrics.write_predictions(out, preds)
    loaded = pd.read_json(out).to_dict()
    assert out.exists()

    full_manifest = load_manifest("A")
    full_preds = {
        "dataset_A": {sid: LABELS["A"][:3] for sid in load_split("A", "test")["sample_id"]},
        "dataset_B": {sid: LABELS["B"][:3] for sid in load_split("B", "test")["sample_id"]},
    }
    manifests = {"dataset_A": full_manifest, "dataset_B": load_manifest("B")}
    errors = metrics.validate_predictions(full_preds, manifests)
    assert errors == []

    broken = {k: dict(v) for k, v in full_preds.items()}
    some_id = next(iter(broken["dataset_A"]))
    broken["dataset_A"][some_id] = [LABELS["A"][0], LABELS["A"][0], LABELS["A"][1]]
    errors = metrics.validate_predictions(broken, manifests)
    assert any("distinct" in e for e in errors)
