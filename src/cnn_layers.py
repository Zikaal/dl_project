import numpy as np

# ---------- utils: im2col / col2im (быстро для conv/pool) ----------
def im2col(x, kH, kW, pad=0, stride=1):
    # x: (N, C, H, W)
    N, C, H, W = x.shape
    H_out = (H + 2*pad - kH)//stride + 1
    W_out = (W + 2*pad - kW)//stride + 1

    x_pad = np.pad(x, ((0,0),(0,0),(pad,pad),(pad,pad)), mode="constant")

    col = np.zeros((N, C, kH, kW, H_out, W_out), dtype=x.dtype)
    for y in range(kH):
        y_max = y + stride*H_out
        for x_ in range(kW):
            x_max = x_ + stride*W_out
            col[:, :, y, x_, :, :] = x_pad[:, :, y:y_max:stride, x_:x_max:stride]

    col = col.transpose(0,4,5,1,2,3).reshape(N*H_out*W_out, -1)  # (N*H_out*W_out, C*kH*kW)
    return col, H_out, W_out

def col2im(col, x_shape, kH, kW, pad=0, stride=1, H_out=None, W_out=None):
    N, C, H, W = x_shape
    if H_out is None or W_out is None:
        H_out = (H + 2*pad - kH)//stride + 1
        W_out = (W + 2*pad - kW)//stride + 1

    col = col.reshape(N, H_out, W_out, C, kH, kW).transpose(0,3,4,5,1,2)
    x_pad = np.zeros((N, C, H + 2*pad, W + 2*pad), dtype=col.dtype)

    for y in range(kH):
        y_max = y + stride*H_out
        for x_ in range(kW):
            x_max = x_ + stride*W_out
            x_pad[:, :, y:y_max:stride, x_:x_max:stride] += col[:, :, y, x_, :, :]

    if pad == 0:
        return x_pad
    return x_pad[:, :, pad:-pad, pad:-pad]


# ---------- activations ----------
def relu_forward(x):
    out = np.maximum(0, x)
    cache = x
    return out, cache

def relu_backward(dout, cache):
    x = cache
    dx = dout * (x > 0)
    return dx

# ---------- affine (fully-connected) ----------
def affine_forward(x, W, b):
    # x: (N, D) -> out: (N, M)
    out = x @ W + b
    cache = (x, W, b)
    return out, cache

def affine_backward(dout, cache):
    x, W, b = cache
    dx = dout @ W.T
    dW = x.T @ dout
    db = dout.sum(axis=0)
    return dx, dW, db

# ---------- softmax + cross-entropy ----------
def softmax_loss(scores, y):
    # scores: (N, C)
    scores = scores - scores.max(axis=1, keepdims=True)
    exp = np.exp(scores)
    probs = exp / exp.sum(axis=1, keepdims=True)

    N = scores.shape[0]
    loss = -np.mean(np.log(probs[np.arange(N), y] + 1e-12))

    dscores = probs.copy()
    dscores[np.arange(N), y] -= 1
    dscores /= N
    return loss, dscores


# ---------- convolution layer ----------
def conv_forward(x, W, b, stride=1, pad=0):
    """
    x: (N, C, H, W)
    W: (F, C, kH, kW)
    b: (F,)
    """
    N, C, H, W_in = x.shape
    F, C2, kH, kW = W.shape
    assert C == C2

    col, H_out, W_out = im2col(x, kH, kW, pad=pad, stride=stride)  # (N*H_out*W_out, C*kH*kW)
    W_col = W.reshape(F, -1).T  # (C*kH*kW, F)

    out = col @ W_col + b  # (N*H_out*W_out, F)
    out = out.reshape(N, H_out, W_out, F).transpose(0,3,1,2)  # (N, F, H_out, W_out)

    cache = (x, W, b, stride, pad, col, W_col, H_out, W_out, kH, kW)
    return out, cache

def conv_backward(dout, cache):
    (x, W, b, stride, pad, col, W_col, H_out, W_out, kH, kW) = cache
    N, C, H, W_in = x.shape
    F = W.shape[0]

    dout_reshaped = dout.transpose(0,2,3,1).reshape(-1, F)  # (N*H_out*W_out, F)

    db = dout_reshaped.sum(axis=0)
    dW_col = col.T @ dout_reshaped  # (C*kH*kW, F)
    dW = dW_col.T.reshape(W.shape)

    dcol = dout_reshaped @ W_col.T  # (N*H_out*W_out, C*kH*kW)
    dx = col2im(dcol, x.shape, kH, kW, pad=pad, stride=stride, H_out=H_out, W_out=W_out)

    return dx, dW, db


# ---------- pooling layer (max / avg) ----------
def pool_forward(x, pool=2, stride=2, mode="max"):
    """
    x: (N, C, H, W)
    """
    N, C, H, W = x.shape
    kH = kW = pool
    pad = 0

    col, H_out, W_out = im2col(x, kH, kW, pad=pad, stride=stride)  # (N*H_out*W_out, C*kH*kW)
    col = col.reshape(N*H_out*W_out, C, kH*kW)  # group by channel

    if mode == "max":
        out = col.max(axis=2)
        argmax = col.argmax(axis=2)
        cache = (x, pool, stride, mode, col, argmax, H_out, W_out)
    elif mode == "avg":
        out = col.mean(axis=2)
        cache = (x, pool, stride, mode, col, None, H_out, W_out)
    else:
        raise ValueError("mode must be 'max' or 'avg'")

    out = out.reshape(N, H_out, W_out, C).transpose(0,3,1,2)  # (N,C,H_out,W_out)
    return out, cache

def pool_backward(dout, cache):
    x, pool, stride, mode, col, argmax, H_out, W_out = cache
    N, C, H, W = x.shape
    kH = kW = pool

    dout_flat = dout.transpose(0,2,3,1).reshape(-1, C)  # (N*H_out*W_out, C)

    dcol = np.zeros((dout_flat.shape[0], C, kH*kW), dtype=x.dtype)

    if mode == "max":
        # put gradient only to max locations
        idx = np.arange(dout_flat.shape[0])[:, None]
        ch  = np.arange(C)[None, :]
        dcol[idx, ch, argmax] = dout_flat
    else:
        # avg: distribute evenly
        dcol += dout_flat[:, :, None] / (kH*kW)

    dcol = dcol.reshape(dout_flat.shape[0], -1)  # (N*H_out*W_out, C*kH*kW)

    dx = col2im(dcol, x.shape, kH, kW, pad=0, stride=stride, H_out=H_out, W_out=W_out)
    return dx
