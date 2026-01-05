import numpy as np

from src.section1_mlp_numpy import (
    init_params,
    forward_classification,
    backward_classification,
    update,
)

from src.mlp_components import softmax, ce_loss


def test_softmax_rows_sum_to_one():
    np.random.seed(0)
    logits = np.random.randn(8, 10).astype(np.float32)
    probs = softmax(logits)
    assert probs.shape == (8, 10)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-6)


def test_forward_classification_shapes_and_cache_keys():
    np.random.seed(0)

    X = np.random.randn(5, 784).astype(np.float32)
    sizes = [784, 32, 16, 10]
    W, b = init_params(sizes, activation="relu")

    probs, cache = forward_classification(X, W, b, activation="relu")

    assert probs.shape == (5, 10)

    # minimal cache checks
    assert "A0" in cache
    assert "probs" in cache
    assert "logits" in cache

    # hidden activations exist
    assert "Z1" in cache and "A1" in cache
    assert "Z2" in cache and "A2" in cache


def test_backward_classification_shapes_match_params():
    np.random.seed(0)

    X = np.random.randn(6, 784).astype(np.float32)
    y = np.random.randint(0, 10, size=(6,), dtype=np.int64)

    sizes = [784, 32, 16, 10]
    W, b = init_params(sizes, activation="tanh")

    probs, cache = forward_classification(X, W, b, activation="tanh")
    dW, db = backward_classification(y, W, cache, activation="tanh")

    assert len(dW) == len(W)
    assert len(db) == len(b)

    for i in range(len(W)):
        assert dW[i].shape == W[i].shape
        assert db[i].shape == b[i].shape


def test_ce_loss_finite():
    np.random.seed(0)
    probs = np.full((4, 10), 0.1, dtype=np.float32)
    y = np.array([0, 1, 2, 3], dtype=np.int64)
    loss = ce_loss(probs, y)
    assert np.isfinite(loss)


def test_one_update_step_not_worse_on_tiny_batch():

    np.random.seed(0)

    X = np.random.randn(64, 784).astype(np.float32)
    y = np.random.randint(0, 10, size=(64,), dtype=np.int64)

    sizes = [784, 64, 10]
    W, b = init_params(sizes, activation="relu")

    probs0, cache0 = forward_classification(X, W, b, activation="relu")
    loss0 = ce_loss(probs0, y)

    dW, db = backward_classification(y, W, cache0, activation="relu")
    W2, b2 = update([w.copy() for w in W], [bb.copy() for bb in b], dW, db, lr=0.01)

    probs1, _ = forward_classification(X, W2, b2, activation="relu")
    loss1 = ce_loss(probs1, y)

    assert loss1 <= loss0 + 1e-6
