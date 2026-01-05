import numpy as np
import random
import matplotlib.pyplot as plt

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

def accuracy(y_true, y_pred):
    return (y_true == y_pred).mean()

def batches(X, y, batch_size=128, shuffle=True):
    idx = np.arange(len(X))
    if shuffle:
        np.random.shuffle(idx)
    for start in range(0, len(X), batch_size):
        b = idx[start:start+batch_size]
        yield X[b], y[b]

def plot_curves(train_list, test_list, title, y_label):
    plt.figure()
    plt.plot(train_list, label="train")
    plt.plot(test_list, label="test")
    plt.title(title)
    plt.xlabel("epoch")
    plt.ylabel(y_label)
    plt.legend()
    plt.show()

def confusion_matrix_np(y_true, y_pred, num_classes: int):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm

def plot_confusion_matrix(cm, class_names, title="Confusion Matrix", normalize=False):
    cm_plot = cm.astype(np.float32)
    if normalize:
        row_sum = cm_plot.sum(axis=1, keepdims=True) + 1e-12
        cm_plot = cm_plot / row_sum

    plt.figure()
    plt.imshow(cm_plot, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=45, ha="right")
    plt.yticks(ticks, class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()

def plot_lr_schedule(lrs, title="Learning Rate Schedule"):
    plt.figure()
    plt.plot(lrs)
    plt.title(title)
    plt.xlabel("epoch")
    plt.ylabel("lr")
    plt.show()

def plot_grad_norms(grad_norms, title="Gradient Magnitude (L2 norm)"):
    plt.figure()
    plt.plot(grad_norms)
    plt.title(title)
    plt.xlabel("step")
    plt.ylabel("||grad||")
    plt.show()

def count_params_numpy(W_list, b_list):
    return int(sum(w.size for w in W_list) + sum(b.size for b in b_list))

def count_params_torch(model):
    import torch
    return int(sum(p.numel() for p in model.parameters()))