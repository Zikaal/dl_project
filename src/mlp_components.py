import numpy as np

# ---------- activations ----------
def sigmoid(z):
    z = np.clip(z, -50, 50)
    return 1 / (1 + np.exp(-z))

def dsigmoid(a):
    return a * (1 - a)

def tanh(z):
    return np.tanh(z)

def dtanh(a):
    return 1 - a*a

def relu(z):
    return np.maximum(0, z)

def drelu(z):
    return (z > 0).astype(np.float32)

def act(z, name):
    if name == "sigmoid":
        return sigmoid(z)
    if name == "tanh":
        return tanh(z)
    if name == "relu":
        return relu(z)
    raise ValueError("activation must be sigmoid/tanh/relu")

def dact(z, a, name):
    if name == "sigmoid":
        return dsigmoid(a)
    if name == "tanh":
        return dtanh(a)
    if name == "relu":
        return drelu(z)
    raise ValueError("activation must be sigmoid/tanh/relu")

# ---------- softmax + cross entropy ----------
def softmax(logits):
    z = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(z)
    return exp / exp.sum(axis=1, keepdims=True)

def ce_loss(probs, y):
    eps = 1e-12
    return -np.mean(np.log(probs[np.arange(len(y)), y] + eps))

def dlogits_softmax_ce(probs, y):
    # (probs - onehot)/N
    N = len(y)
    grad = probs.copy()
    grad[np.arange(N), y] -= 1
    grad /= N
    return grad

# ---------- regression (MSE) ----------
def mse_loss(y_pred, y_true):
    return np.mean((y_pred - y_true)**2)

def dmse(y_pred, y_true):
    return 2*(y_pred - y_true)/len(y_true)
