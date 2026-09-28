import numpy as np

from hw1.layermix import fit_layermix


def test_learns_the_informative_layer():
    rng = np.random.default_rng(0)
    n, layers, dim = 300, 4, 8
    y = rng.integers(0, 6, n)
    X = rng.normal(size=(n, layers, dim)).astype(np.float32)
    X[np.arange(n), 2, y] += 4.0
    fitted = fit_layermix(X, y, epochs=400, lr=1e-2, device="cpu")
    w = fitted.layer_weights()
    assert w.argmax() == 2 and abs(w.sum() - 1) < 1e-5
    probs = fitted.predict_proba(X)
    assert probs.shape == (n, 6)
    assert (probs.argmax(1) == y).mean() > 0.8
