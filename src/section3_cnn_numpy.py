import numpy as np
import matplotlib.pyplot as plt

from src.cnn_layers import (
    conv_forward, conv_backward,
    relu_forward, relu_backward,
    pool_forward, pool_backward,
    affine_forward, affine_backward,
    softmax_loss
)

# ---------------------------
# CNN: Conv-ReLU-Pool -> Conv-ReLU-Pool -> FC -> Softmax
# ---------------------------
def init_cnn(seed=42):
    np.random.seed(seed)

    # architecture:
    # conv1: 1 -> 8 filters 3x3 pad=1
    # conv2: 8 -> 16 filters 3x3 pad=1
    # after two pools: 28->14->7, channels=16 => 16*7*7=784

    params = {}

    # conv1
    params["W1"] = (0.1*np.random.randn(8, 1, 3, 3)).astype(np.float32)
    params["b1"] = np.zeros((8,), dtype=np.float32)

    # conv2
    params["W2"] = (0.1*np.random.randn(16, 8, 3, 3)).astype(np.float32)
    params["b2"] = np.zeros((16,), dtype=np.float32)

    # fc
    params["W3"] = (0.1*np.random.randn(16*7*7, 10)).astype(np.float32)
    params["b3"] = np.zeros((10,), dtype=np.float32)

    return params

def cnn_forward(X, params, pool_mode="max"):
    caches = {}

    # conv1
    c1, cache_c1 = conv_forward(X, params["W1"], params["b1"], stride=1, pad=1)
    r1, cache_r1 = relu_forward(c1)
    p1, cache_p1 = pool_forward(r1, pool=2, stride=2, mode=pool_mode)  # 28->14

    # conv2
    c2, cache_c2 = conv_forward(p1, params["W2"], params["b2"], stride=1, pad=1)
    r2, cache_r2 = relu_forward(c2)
    p2, cache_p2 = pool_forward(r2, pool=2, stride=2, mode=pool_mode)  # 14->7

    # flatten
    N = X.shape[0]
    flat = p2.reshape(N, -1)

    scores, cache_fc = affine_forward(flat, params["W3"], params["b3"])

    caches["c1"]=cache_c1; caches["r1"]=cache_r1; caches["p1"]=cache_p1
    caches["c2"]=cache_c2; caches["r2"]=cache_r2; caches["p2"]=cache_p2
    caches["flat_shape"]=p2.shape
    caches["fc"]=cache_fc

    # also save activations for visualization
    caches["act_c1"] = c1
    caches["act_r1"] = r1
    caches["act_p1"] = p1
    caches["act_c2"] = c2
    caches["act_r2"] = r2
    caches["act_p2"] = p2

    return scores, caches

def cnn_backward(dscores, params, caches, pool_mode="max"):
    grads = {}

    # fc backward
    dflat, dW3, db3 = affine_backward(dscores, caches["fc"])
    grads["W3"]=dW3; grads["b3"]=db3

    # reshape back
    p2_shape = caches["flat_shape"]
    dp2 = dflat.reshape(p2_shape)

    # pool2 backward
    dr2 = pool_backward(dp2, caches["p2"])
    dc2 = relu_backward(dr2, caches["r2"])

    # conv2 backward
    dp1, dW2, db2 = conv_backward(dc2, caches["c2"])
    grads["W2"]=dW2; grads["b2"]=db2

    # pool1 backward
    dr1 = pool_backward(dp1, caches["p1"])
    dc1 = relu_backward(dr1, caches["r1"])

    # conv1 backward
    dX, dW1, db1 = conv_backward(dc1, caches["c1"])
    grads["W1"]=dW1; grads["b1"]=db1

    return grads

def sgd_update(params, grads, lr=0.01):
    for k in params:
        params[k] -= lr * grads[k]
    return params

def accuracy(pred, y):
    return float((pred == y).mean())

def iterate_minibatches(X, y, batch_size=128, shuffle=True):
    idx = np.arange(len(X))
    if shuffle:
        np.random.shuffle(idx)
    for start in range(0, len(X), batch_size):
        b = idx[start:start+batch_size]
        yield X[b], y[b]

def train_cnn(X_train, y_train, X_test, y_test,
              epochs=5, batch_size=128, lr=0.01,
              pool_mode="max", seed=42):

    params = init_cnn(seed=seed)
    history = {"train_loss": [], "test_loss": [], "train_acc": [], "test_acc": []}

    for ep in range(1, epochs+1):
        losses = []
        correct = 0
        total = 0

        for Xb, yb in iterate_minibatches(X_train, y_train, batch_size=batch_size, shuffle=True):
            scores, caches = cnn_forward(Xb, params, pool_mode=pool_mode)
            loss, dscores = softmax_loss(scores, yb)

            grads = cnn_backward(dscores, params, caches, pool_mode=pool_mode)
            params = sgd_update(params, grads, lr=lr)

            losses.append(loss)
            pred = scores.argmax(axis=1)
            correct += int((pred == yb).sum())
            total += len(yb)

        train_loss = float(np.mean(losses))
        train_acc = float(correct / total)

        # test
        scores_te, _ = cnn_forward(X_test, params, pool_mode=pool_mode)
        test_loss, _ = softmax_loss(scores_te, y_test)
        test_acc = accuracy(scores_te.argmax(axis=1), y_test)

        history["train_loss"].append(train_loss)
        history["test_loss"].append(float(test_loss))
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)

        print(f"[pool={pool_mode}] Epoch {ep}/{epochs} "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
              f"test_acc={test_acc:.4f}")

    return params, history

def plot_history(history, title_prefix="CNN"):
    plt.figure()
    plt.plot(history["train_loss"], label="train")
    plt.plot(history["test_loss"], label="test")
    plt.title(f"{title_prefix} Loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(history["train_acc"], label="train")
    plt.plot(history["test_acc"], label="test")
    plt.title(f"{title_prefix} Accuracy")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend()
    plt.show()


# ---------------------------
# 2) Filter Visualization
# ---------------------------
def visualize_conv1_filters(params, max_filters=8):
    W1 = params["W1"]  # (8,1,3,3)
    F = min(W1.shape[0], max_filters)

    plt.figure(figsize=(2*F, 2))
    for i in range(F):
        plt.subplot(1, F, i+1)
        plt.imshow(W1[i, 0], cmap="gray")
        plt.title(f"f{i}")
        plt.axis("off")
    plt.suptitle("Conv1 learned filters (3x3)")
    plt.show()

def visualize_feature_maps(X_one, params, pool_mode="max", layer="conv1", n_maps=8):
    # X_one: (1,1,28,28)
    scores, caches = cnn_forward(X_one, params, pool_mode=pool_mode)

    if layer == "conv1":
        maps = caches["act_c1"][0]  # (8,28,28)
        title = "Conv1 responses"
    elif layer == "relu1":
        maps = caches["act_r1"][0]
        title = "ReLU1 responses"
    elif layer == "pool1":
        maps = caches["act_p1"][0]  # (8,14,14)
        title = "Pool1 responses"
    elif layer == "conv2":
        maps = caches["act_c2"][0]  # (16,14,14)
        title = "Conv2 responses"
    elif layer == "pool2":
        maps = caches["act_p2"][0]  # (16,7,7)
        title = "Pool2 responses"
    else:
        raise ValueError("layer must be conv1/relu1/pool1/conv2/pool2")

    k = min(n_maps, maps.shape[0])
    plt.figure(figsize=(2*k, 2))
    for i in range(k):
        plt.subplot(1, k, i+1)
        plt.imshow(maps[i], cmap="gray")
        plt.axis("off")
        plt.title(f"m{i}")
    plt.suptitle(title)
    plt.show()


# ---------------------------
# 3) Receptive Field (theoretical)
# ---------------------------
def receptive_field(layers):
    """
    layers: list of dicts:
      {"type":"conv","k":3,"s":1}
      {"type":"pool","k":2,"s":2}
    Returns RF size and effective stride (jump).
    """
    rf = 1      # receptive field
    j = 1       # jump (effective stride)
    for L in layers:
        k = L["k"]
        s = L["s"]
        rf = rf + (k - 1) * j
        j = j * s
    return rf, j

def default_cnn_receptive_field():
    layers = [
        {"type":"conv","k":3,"s":1},
        {"type":"pool","k":2,"s":2},
        {"type":"conv","k":3,"s":1},
        {"type":"pool","k":2,"s":2},
    ]
    rf, j = receptive_field(layers)
    print(f"Theoretical receptive field after pool2: {rf}x{rf}, effective stride={j}")
    return rf, j


# ---------------------------
# 4) Pooling Comparison
# ---------------------------
def compare_pooling(X_train, y_train, X_test, y_test, epochs=5, lr=0.01):
    results = {}
    for mode in ["max", "avg"]:
        params, hist = train_cnn(X_train, y_train, X_test, y_test,
                                 epochs=epochs, batch_size=128, lr=lr,
                                 pool_mode=mode, seed=42)
        results[mode] = (params, hist)

    # plot test acc together
    plt.figure()
    for mode in results:
        plt.plot(results[mode][1]["test_acc"], label=mode)
    plt.title("Test Accuracy: Max Pool vs Avg Pool")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend()
    plt.show()

    return results
