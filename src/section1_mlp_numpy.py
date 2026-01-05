import numpy as np
import matplotlib.pyplot as plt

from src.mlp_components import act, dact, softmax, ce_loss, dlogits_softmax_ce, mse_loss, dmse
from src.utils import batches, accuracy, plot_curves


# =========================================================
# Part 1: MLP for classification (MNIST)
# =========================================================

def init_params(sizes, activation="relu"):
    # sizes: [input, h1, h2, ..., output]
    W, b = [], []
    for i in range(len(sizes)-1):
        fan_in = sizes[i]
        if activation == "relu":
            scale = np.sqrt(2.0/fan_in)
        else:
            scale = np.sqrt(1.0/fan_in)
        W.append((np.random.randn(sizes[i], sizes[i+1]) * scale).astype(np.float32))
        b.append(np.zeros((sizes[i+1],), dtype=np.float32))
    return W, b

def forward_classification(X, W, b, activation):
    cache = {"A0": X}
    A = X
    # hidden layers
    for i in range(len(W)-1):
        Z = A @ W[i] + b[i]
        A = act(Z, activation)
        cache[f"Z{i+1}"] = Z
        cache[f"A{i+1}"] = A
    # output
    logits = A @ W[-1] + b[-1]
    probs = softmax(logits)
    cache["logits"] = logits
    cache["probs"] = probs
    return probs, cache

def backward_classification(y, W, cache, activation):
    probs = cache["probs"]
    dlogits = dlogits_softmax_ce(probs, y)

    dW = [None]*len(W)
    db = [None]*len(W)

    # last layer
    A_last = cache[f"A{len(W)-1}"]
    dW[-1] = A_last.T @ dlogits
    db[-1] = dlogits.sum(axis=0)
    dA = dlogits @ W[-1].T

    # hidden layers
    for i in reversed(range(len(W)-1)):
        Z = cache[f"Z{i+1}"]
        A = cache[f"A{i+1}"]
        dZ = dA * dact(Z, A, activation)

        A_prev = cache[f"A{i}"]
        dW[i] = A_prev.T @ dZ
        db[i] = dZ.sum(axis=0)
        dA = dZ @ W[i].T

    return dW, db

def update(W, b, dW, db, lr):
    for i in range(len(W)):
        W[i] -= lr*dW[i]
        b[i] -= lr*db[i]
    return W, b

def train_mnist_mlp(X_train, y_train, X_test, y_test,
                    sizes=[784, 256, 128, 10],
                    activation="relu",
                    epochs=10, batch_size=128, lr=0.08):

    W, b = init_params(sizes, activation)
    train_loss_list, test_loss_list = [], []
    train_acc_list, test_acc_list = [], []

    for ep in range(1, epochs+1):
        # --- train ---
        losses = []
        correct = 0
        total = 0

        for Xb, yb in batches(X_train, y_train, batch_size=batch_size, shuffle=True):
            probs, cache = forward_classification(Xb, W, b, activation)
            loss = ce_loss(probs, yb)
            dW, db = backward_classification(yb, W, cache, activation)
            W, b = update(W, b, dW, db, lr)

            losses.append(loss)
            pred = probs.argmax(axis=1)
            correct += (pred == yb).sum()
            total += len(yb)

        train_loss = float(np.mean(losses))
        train_acc = float(correct/total)

        # --- test ---
        probs_test, _ = forward_classification(X_test, W, b, activation)
        test_loss = float(ce_loss(probs_test, y_test))
        test_pred = probs_test.argmax(axis=1)
        test_acc = float(accuracy(y_test, test_pred))

        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)

        print(f"Epoch {ep}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} | test_acc={test_acc:.4f}")

    history = {
        "train_loss": train_loss_list,
        "test_loss": test_loss_list,
        "train_acc": train_acc_list,
        "test_acc": test_acc_list,
    }
    return W, b, history


def plot_activation_histograms(X_sample, W, b, activation):
    # run forward once and plot A1, A2 distributions
    _, cache = forward_classification(X_sample, W, b, activation)

    keys = [k for k in cache.keys() if k.startswith("A") and k != "A0"]
    keys = sorted(keys, key=lambda s: int(s[1:]))

    for k in keys:
        a = cache[k].ravel()
        plt.figure()
        plt.hist(a, bins=60)
        plt.title(f"{k} distribution ({activation})")
        plt.xlabel("activation value")
        plt.ylabel("count")
        plt.show()


# =========================================================
# Part 2: Activation comparison on toy dataset (XOR)
# =========================================================
def make_xor(n=800, noise=0.12, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, size=(n, 2)).astype(np.float32)
    y = ((X[:, 0]*X[:, 1]) > 0).astype(np.int64)
    X += rng.normal(0, noise, size=X.shape).astype(np.float32)
    return X, y

def run_activation_comparison(epochs=60, lr=0.08):
    X, y = make_xor()

    # split 80/20
    idx = np.arange(len(X))
    np.random.shuffle(idx)
    split = int(0.8*len(X))
    tr, te = idx[:split], idx[split:]
    X_train, y_train = X[tr], y[tr]
    X_test, y_test = X[te], y[te]

    results = {}

    for act_name in ["sigmoid", "tanh", "relu"]:
        W, b = init_params([2, 32, 32, 2], activation=act_name)
        tr_loss, te_loss, te_acc = [], [], []

        for ep in range(epochs):
            # train
            for Xb, yb in batches(X_train, y_train, batch_size=64, shuffle=True):
                probs, cache = forward_classification(Xb, W, b, act_name)
                dW, db = backward_classification(yb, W, cache, act_name)
                W, b = update(W, b, dW, db, lr)

            # eval
            probs_tr, _ = forward_classification(X_train, W, b, act_name)
            probs_te, _ = forward_classification(X_test, W, b, act_name)
            tr_loss.append(ce_loss(probs_tr, y_train))
            te_loss.append(ce_loss(probs_te, y_test))
            te_acc.append(accuracy(y_test, probs_te.argmax(axis=1)))

        results[act_name] = {"train_loss": tr_loss, "test_loss": te_loss, "test_acc": te_acc}
        print(f"{act_name}: final test_acc = {te_acc[-1]:.4f}")

        # activation distributions
        plot_activation_histograms(X_test[:200], W, b, act_name)

    # plots
    plt.figure()
    for act_name in results:
        plt.plot(results[act_name]["test_loss"], label=f"{act_name}")
    plt.title("Toy XOR: Test Loss Comparison")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.show()

    plt.figure()
    for act_name in results:
        plt.plot(results[act_name]["test_acc"], label=f"{act_name}")
    plt.title("Toy XOR: Test Accuracy Comparison")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend()
    plt.show()

    return results


# =========================================================
# Part 3: Universal approximation (sin) - regression MLP
# =========================================================
def make_sin(n=1600):
    x = np.linspace(-2*np.pi, 2*np.pi, n).astype(np.float32).reshape(-1, 1)
    y = np.sin(x).astype(np.float32)
    return x, y

def forward_regression(X, W, b, activation):
    cache = {"A0": X}
    A = X
    for i in range(len(W)-1):
        Z = A @ W[i] + b[i]
        A = act(Z, activation)
        cache[f"Z{i+1}"] = Z
        cache[f"A{i+1}"] = A
    y_pred = A @ W[-1] + b[-1]  # linear output
    cache["y_pred"] = y_pred
    return y_pred, cache

def backward_regression(y_true, W, cache, activation):
    y_pred = cache["y_pred"]
    dy = dmse(y_pred, y_true)

    dW = [None]*len(W)
    db = [None]*len(W)

    A_last = cache[f"A{len(W)-1}"]
    dW[-1] = A_last.T @ dy
    db[-1] = dy.sum(axis=0)
    dA = dy @ W[-1].T

    for i in reversed(range(len(W)-1)):
        Z = cache[f"Z{i+1}"]
        A = cache[f"A{i+1}"]
        dZ = dA * dact(Z, A, activation)
        A_prev = cache[f"A{i}"]
        dW[i] = A_prev.T @ dZ
        db[i] = dZ.sum(axis=0)
        dA = dZ @ W[i].T

    return dW, db

def train_sin_demo():
    X, y = make_sin()
    idx = np.arange(len(X))
    np.random.shuffle(idx)
    split = int(0.8*len(X))
    tr, te = idx[:split], idx[split:]
    X_train, y_train = X[tr], y[tr]
    X_test, y_test = X[te], y[te]

    configs = [
        ("small",  [1, 16, 1]),
        ("medium", [1, 64, 1]),
        ("big",    [1, 128, 128, 1]),
    ]

    final_mse = {}

    for name, sizes in configs:
        W, b = init_params(sizes, activation="tanh")
        train_losses, test_losses = [], []

        for ep in range(1500):
            for Xb, yb in batches(X_train, y_train, batch_size=128, shuffle=True):
                y_pred, cache = forward_regression(Xb, W, b, "tanh")
                dW, db = backward_regression(yb, W, cache, "tanh")
                W, b = update(W, b, dW, db, lr=0.01)

            pred_tr, _ = forward_regression(X_train, W, b, "tanh")
            pred_te, _ = forward_regression(X_test, W, b, "tanh")
            train_losses.append(mse_loss(pred_tr, y_train))
            test_losses.append(mse_loss(pred_te, y_test))

        final_mse[name] = test_losses[-1]

        # plot loss
        plt.figure()
        plt.plot(train_losses, label="train")
        plt.plot(test_losses, label="test")
        plt.title(f"Sin Approximation Loss: {name} {sizes}")
        plt.xlabel("epoch")
        plt.ylabel("MSE")
        plt.legend()
        plt.show()

        # plot prediction vs true
        order = np.argsort(X_test[:, 0])
        Xs = X_test[order]
        ys = y_test[order]
        yp, _ = forward_regression(Xs, W, b, "tanh")

        plt.figure()
        plt.plot(Xs[:, 0], ys[:, 0], label="true sin(x)")
        plt.plot(Xs[:, 0], yp[:, 0], label="pred")
        plt.title(f"Approximation Quality: {name}")
        plt.legend()
        plt.show()

    # bar chart MSE vs size
    plt.figure()
    plt.bar(list(final_mse.keys()), list(final_mse.values()))
    plt.title("Test MSE vs Network Size")
    plt.ylabel("MSE")
    plt.show()

    print("Final MSE:", final_mse)
