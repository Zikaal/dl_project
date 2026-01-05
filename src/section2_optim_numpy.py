import numpy as np
import matplotlib.pyplot as plt

from src.mlp_components import act, dact   
from src.utils import batches, accuracy


# =========================================================
# Stable softmax + cross-entropy WITHOUT forcing float32
# (important for gradient checking)
# =========================================================
def softmax_stable(logits: np.ndarray) -> np.ndarray:
    # keep dtype of logits (float64 during grad check)
    z = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(z)
    return exp / np.sum(exp, axis=1, keepdims=True)

def ce_loss_stable(probs: np.ndarray, y: np.ndarray) -> float:
    eps = 1e-12
    y = y.astype(np.int64)
    n = probs.shape[0]
    return float(-np.mean(np.log(probs[np.arange(n), y] + eps)))

def dlogits_softmax_ce_stable(probs: np.ndarray, y: np.ndarray) -> np.ndarray:
    # (probs - onehot)/N
    y = y.astype(np.int64)
    n = probs.shape[0]
    grad = probs.copy()
    grad[np.arange(n), y] -= 1.0
    grad /= n
    return grad


# =========================================================
# 1) Two-layer network (1 hidden + output) + backprop
# =========================================================
def init_two_layer(input_dim, hidden_dim, num_classes, activation="relu", seed=42):
    np.random.seed(seed)

    if activation == "relu":
        s1 = np.sqrt(2.0 / input_dim)
    else:
        s1 = np.sqrt(1.0 / input_dim)

    W1 = (np.random.randn(input_dim, hidden_dim) * s1).astype(np.float32)
    b1 = np.zeros((hidden_dim,), dtype=np.float32)

    s2 = np.sqrt(1.0 / hidden_dim)
    W2 = (np.random.randn(hidden_dim, num_classes) * s2).astype(np.float32)
    b2 = np.zeros((num_classes,), dtype=np.float32)

    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}


def forward_two_layer(X, params, activation):
    W1, b1, W2, b2 = params["W1"], params["b1"], params["W2"], params["b2"]

    Z1 = X @ W1 + b1
    A1 = act(Z1, activation)
    Z2 = A1 @ W2 + b2
    probs = softmax_stable(Z2)

    cache = {"X": X, "Z1": Z1, "A1": A1, "Z2": Z2, "probs": probs}
    return probs, cache


def backward_two_layer(y, params, cache, activation):
    W2 = params["W2"]
    X, Z1, A1, probs = cache["X"], cache["Z1"], cache["A1"], cache["probs"]

    dZ2 = dlogits_softmax_ce_stable(probs, y)   # (N,C)

    dW2 = A1.T @ dZ2
    db2 = np.sum(dZ2, axis=0)

    dA1 = dZ2 @ W2.T
    dZ1 = dA1 * dact(Z1, A1, activation)

    dW1 = X.T @ dZ1
    db1 = np.sum(dZ1, axis=0)

    return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}


def compute_loss_and_grads(X, y, params, activation):
    probs, cache = forward_two_layer(X, params, activation)
    loss = ce_loss_stable(probs, y)
    grads = backward_two_layer(y, params, cache, activation)
    return loss, grads


# =========================================================
# 2) Gradient checking (numerical gradient) - WORKING
# =========================================================
def relative_error_scalar(a, b, eps=1e-12):
    return abs(a - b) / max(eps, abs(a) + abs(b))


def grad_check(params, X, y, activation="tanh", h=1e-5, num_checks=20, seed=42):
    """
    Uses float64 internally so finite-differences are stable.
    Returns {param_name: max_relative_error}.
    """
    np.random.seed(seed)

    # float64 copies
    params64 = {k: params[k].astype(np.float64).copy() for k in params}
    X64 = X.astype(np.float64)

    _, grads = compute_loss_and_grads(X64, y, params64, activation)
    grads = {k: grads[k].astype(np.float64) for k in grads}

    report = {}

    for key in params64:
        P = params64[key]
        G = grads[key]
        max_rel = 0.0

        for _ in range(num_checks):
            idx = tuple(np.random.randint(s) for s in P.shape)
            old = P[idx]

            P[idx] = old + h
            loss_plus, _ = compute_loss_and_grads(X64, y, params64, activation)

            P[idx] = old - h
            loss_minus, _ = compute_loss_and_grads(X64, y, params64, activation)

            P[idx] = old

            grad_num = (loss_plus - loss_minus) / (2.0 * h)
            grad_ana = G[idx]

            rel = relative_error_scalar(grad_num, grad_ana)
            max_rel = max(max_rel, rel)

        report[key] = float(max_rel)

    return report


# =========================================================
# 3) Optimizers: SGD, RMSprop, Adam
# =========================================================
def sgd_update(params, grads, lr):
    for k in params:
        params[k] -= lr * grads[k]
    return params


def rmsprop_init(params):
    return {k: np.zeros_like(params[k]) for k in params}


def rmsprop_update(params, grads, cache, lr, beta=0.9, eps=1e-8):
    for k in params:
        cache[k] = beta * cache[k] + (1 - beta) * (grads[k] ** 2)
        params[k] -= lr * grads[k] / (np.sqrt(cache[k]) + eps)
    return params, cache


def adam_init(params):
    m = {k: np.zeros_like(params[k]) for k in params}
    v = {k: np.zeros_like(params[k]) for k in params}
    return m, v, 0


def adam_update(params, grads, m, v, t, lr, beta1=0.9, beta2=0.999, eps=1e-8):
    t += 1
    for k in params:
        m[k] = beta1 * m[k] + (1 - beta1) * grads[k]
        v[k] = beta2 * v[k] + (1 - beta2) * (grads[k] ** 2)

        m_hat = m[k] / (1 - beta1 ** t)
        v_hat = v[k] / (1 - beta2 ** t)

        params[k] -= lr * m_hat / (np.sqrt(v_hat) + eps)

    return params, m, v, t


# =========================================================
# LR schedules
# =========================================================
def lr_constant(lr0, epoch): 
    return lr0

def lr_step_decay(lr0, epoch, step=5, gamma=0.5):
    return lr0 * (gamma ** (epoch // step))

def lr_exponential(lr0, epoch, gamma=0.97):
    return lr0 * (gamma ** epoch)


# =========================================================
# Gradient magnitude helper (Section 2 requirement)
# =========================================================
def grad_magnitude(grads: dict) -> float:
    """
    Single scalar per update step.
    Here: sum of L2 norms over all parameter gradients.
    """
    gnorm = 0.0
    for g in grads.values():
        gnorm += float(np.linalg.norm(g))
    return gnorm


# =========================================================
# Training + comparison
# =========================================================
def train_two_layer(
    X_train, y_train, X_test, y_test,
    hidden_dim=128,
    activation="relu",
    optimizer="sgd",
    lr0=0.01,
    epochs=10,
    batch_size=128,
    schedule="constant",
    seed=42
):
    params = init_two_layer(X_train.shape[1], hidden_dim, 10, activation=activation, seed=seed)

    if optimizer == "rmsprop":
        cache = rmsprop_init(params)
    elif optimizer == "adam":
        m, v, t = adam_init(params)

    hist = {
        "train_loss": [], "test_loss": [],
        "train_acc": [], "test_acc": [],
        "lr": [],
        "grad_norm": []  
    }

    for ep in range(epochs):
        if schedule == "constant":
            lr = lr_constant(lr0, ep)
        elif schedule == "step":
            lr = lr_step_decay(lr0, ep)
        elif schedule == "exp":
            lr = lr_exponential(lr0, ep)
        else:
            raise ValueError("schedule must be constant/step/exp")

        hist["lr"].append(lr)

        losses = []
        correct = 0
        total = 0

        # batches over train
        for Xb, yb in batches(X_train, y_train, batch_size=batch_size, shuffle=True):
            probs, cache_f = forward_two_layer(Xb, params, activation)
            loss = ce_loss_stable(probs, yb)
            grads = backward_two_layer(yb, params, cache_f, activation)

            hist["grad_norm"].append(grad_magnitude(grads))

            # update step
            if optimizer == "sgd":
                params = sgd_update(params, grads, lr)
            elif optimizer == "rmsprop":
                params, cache = rmsprop_update(params, grads, cache, lr)
            elif optimizer == "adam":
                params, m, v, t = adam_update(params, grads, m, v, t, lr)
            else:
                raise ValueError("optimizer must be sgd/rmsprop/adam")

            losses.append(loss)
            pred = np.argmax(probs, axis=1)
            correct += int((pred == yb).sum())
            total += len(yb)

        train_loss = float(np.mean(losses))
        train_acc = float(correct / total)

        probs_te, _ = forward_two_layer(X_test, params, activation)
        test_loss = float(ce_loss_stable(probs_te, y_test))
        test_acc = float(accuracy(y_test, np.argmax(probs_te, axis=1)))

        hist["train_loss"].append(train_loss)
        hist["test_loss"].append(test_loss)
        hist["train_acc"].append(train_acc)
        hist["test_acc"].append(test_acc)

        print(f"[{optimizer}|{schedule}] Epoch {ep+1}/{epochs} lr={lr:.5f} "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} test_acc={test_acc:.4f}")

    return params, hist


def compare_optimizers(
    X_train, y_train, X_test, y_test,
    activation="relu",
    hidden_dim=128,
    epochs=12,
    lr0=0.01,
    schedule="constant",
    batch_size=128
):
    results = {}
    for opt in ["sgd", "rmsprop", "adam"]:
        _, hist = train_two_layer(
            X_train, y_train, X_test, y_test,
            hidden_dim=hidden_dim,
            activation=activation,
            optimizer=opt,
            lr0=lr0,
            epochs=epochs,
            batch_size=batch_size,
            schedule=schedule
        )
        results[opt] = hist

    # loss
    plt.figure()
    for opt in results:
        plt.plot(results[opt]["test_loss"], label=opt)
    plt.title(f"Test Loss: SGD vs RMSprop vs Adam (schedule={schedule})")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.show()

    # acc
    plt.figure()
    for opt in results:
        plt.plot(results[opt]["test_acc"], label=opt)
    plt.title(f"Test Accuracy: SGD vs RMSprop vs Adam (schedule={schedule})")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend()
    plt.show()

    plt.figure()
    for opt in results:
        plt.plot(results[opt]["grad_norm"], label=opt)
    plt.title(f"Gradient Magnitude per update step (schedule={schedule})")
    plt.xlabel("update step")
    plt.ylabel("sum ||grad||")
    plt.legend()
    plt.show()

    return results
