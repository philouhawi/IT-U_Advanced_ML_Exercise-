import numpy as np
import pandas as pd

# activation & loss functions

def relu(z):
    return np.maximum(0.0, z)

def softmax(z):
    z = z - z.max(axis=0, keepdims=True)
    exp_x = np.exp(z)
    return exp_x / exp_x.sum(axis=0, keepdims=True)


def cross_entropy(P, Y, eps=1e-10):
    """Mean cross-entropy. P: predicted probs (K,N), Y: one-hot targets (K,N)."""
    return -np.sum(Y * np.log(P + eps)) / Y.shape[1]

# multi-layer perceptron 
class MLP:
    def __init__(self, sizes, seed=0):
        rng = np.random.default_rng(seed)
        self.L = len(sizes) - 1
        # He initialisation (suited to ReLU)
        self.W = [rng.normal(0, np.sqrt(2.0 / sizes[i]), (sizes[i + 1], sizes[i]))
                  for i in range(self.L)]
        self.b = [np.zeros((sizes[i + 1], 1)) for i in range(self.L)]

        # Adam state
        self.mW = [np.zeros_like(w) for w in self.W]
        self.vW = [np.zeros_like(w) for w in self.W]
        self.mb = [np.zeros_like(b) for b in self.b]
        self.vb = [np.zeros_like(b) for b in self.b]
        self.t = 0


    # forward: z_l = W_l a_{l-1} + b_l ; a_l = relu(z_l) ; last layer -> softmax
    def forward(self, X):
        self.A = [X]
        self.Z = []
        a = X
        for l in range(self.L):
            z = self.W[l] @ a + self.b[l]
            a = softmax(z) if l == self.L - 1 else relu(z)
            self.Z.append(z)
            self.A.append(a)
        return a
 
    # backprop: returns gradients dW, db for every layer
    def backward(self, Y):
        N = Y.shape[1]
        dW = [None] * self.L
        db = [None] * self.L
        dZ = (self.A[-1] - Y) / N                    # softmax + CE shortcut: (p - y)/N
        for l in reversed(range(self.L)):
            dW[l] = dZ @ self.A[l].T                 # dL/dW_l = delta_l a_{l-1}^T
            db[l] = dZ.sum(axis=1, keepdims=True)    # dL/db_l = sum of delta_l over batch
            if l > 0:
                dA = self.W[l].T @ dZ                # push error to previous layer
                dZ = dA * (self.Z[l - 1] > 0)        # times relu'(z_{l-1})
        return dW, db
 
    # Adam update
    def adam_step(self, dW, db, lr=1e-2, beta1=0.9, beta2=0.999, eps=1e-8):
        self.t += 1
        for l in range(self.L):
            for p, g, m, v in ((self.W, dW, self.mW, self.vW),
                               (self.b, db, self.mb, self.vb)):
                m[l] = beta1 * m[l] + (1 - beta1) * g[l]
                v[l] = beta2 * v[l] + (1 - beta2) * g[l] ** 2
                m_hat = m[l] / (1 - beta1 ** self.t)     # bias correction
                v_hat = v[l] / (1 - beta2 ** self.t)
                p[l] -= lr * m_hat / (np.sqrt(v_hat) + eps)
 
    def predict(self, X):
        return self.forward(X).argmax(axis=0)
 
# gradient check (finite differences vs backprop) 

def gradient_check(seed=0):
    rng = np.random.default_rng(seed)
    net = MLP([4, 5, 3], seed=seed)
    X = rng.normal(size=(4, 7))
    Y = np.eye(3)[rng.integers(0, 3, 7)].T
    net.forward(X)
    dW, _ = net.backward(Y)
    h, worst = 1e-6, 0.0
    for l in range(net.L):
        for idx in np.ndindex(net.W[l].shape):
            old = net.W[l][idx]
            net.W[l][idx] = old + h; lp = cross_entropy(net.forward(X), Y)
            net.W[l][idx] = old - h; lm = cross_entropy(net.forward(X), Y)
            net.W[l][idx] = old
            num = (lp - lm) / (2 * h)
            worst = max(worst, abs(num - dW[l][idx]) / max(1e-8, abs(num) + abs(dW[l][idx])))
    return worst
 
# flight data: airline prediction
def load_airline_data(src_path, val_size = 0.15, test_size = 0.15, seed=0):
    df = pd.read_csv(src_path).drop('Unnamed: 0', axis=1)
    # 'flight' starts with the airline code (e.g. SG-8709), so it would leak the answer
    df = df.drop('flight', axis=1)
    names, y = np.unique(df['airline'], return_inverse=True)                  # y: (N,)
    X = pd.get_dummies(df.drop('airline', axis=1), dtype=float).to_numpy().T  # (features, N)
    Y = np.eye(len(names))[y].T                                               # one-hot, shape (K, N)

    idx = np.random.default_rng(seed).permutation(X.shape[1])
    n_val = int(val_size * X.shape[1])
    n_test = int(test_size * X.shape[1])
    val, test, train = idx[:n_val], idx[n_val:n_val + n_test], idx[n_val + n_test:]

    # scale with training statistics only
    X_mean = X[:, train].mean(axis=1, keepdims=True)
    X_std = X[:, train].std(axis=1, keepdims=True)
    X = (X - X_mean) / X_std

    return (X[:, train], Y[:, train], y[train], 
            X[:, val], Y[:, val], y[val], 
            X[:, test], y[test], names)



def train(net, X, Y, X_val, Y_val, y_val, epochs=1000, batch_size=1024, lr=1e-3, seed=0):
    rng = np.random.default_rng(seed)
    best_loss, best_epoch, best_W, best_b = np.inf, 0, None, None
    for epoch in range(1, epochs + 1):
        idx = rng.permutation(X.shape[1])
        for start in range(0, X.shape[1], batch_size):
            batch = idx[start:start + batch_size]
            net.forward(X[:, batch])
            dW, db = net.backward(Y[:, batch])
            net.adam_step(dW, db, lr=lr)
        train_loss = cross_entropy(net.forward(X), Y)
        val_loss = cross_entropy(net.forward(X_val), Y_val)
        val_acc = (net.predict(X_val) == y_val).mean()
        print(f"epoch {epoch:3d}  train loss {train_loss:.4f}  val loss {val_loss:.4f}  val acc {val_acc:.3f}")
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch
            best_W = [w.copy() for w in net.W]
            best_b = [b.copy() for b in net.b]
    net.W, net.b = best_W, best_b
    print(f"best epoch {best_epoch}  val loss {best_loss:.4f}")

def run(src_path):
    (X_train, Y_train, y_train, 
     X_val, Y_val, y_val, 
     X_test, y_test, names) = load_airline_data(src_path)
    print(f"samples: train {X_train.shape[1]}  val {X_val.shape[1]}  test {X_test.shape[1]}")
    net = MLP([X_train.shape[0], 64, 64, len(names)])
    train(net, X_train, Y_train, X_val, Y_val, y_val)

    y_pred = net.predict(X_test)
    majority = np.bincount(y_train).argmax()
    print(f"test accuracy {(y_pred == y_test).mean():.3f}")
    print(f"baseline, always {names[majority]}: {(y_test == majority).mean():.3f}")
    for k, name in enumerate(names):
        print(f"{name:10s} recall {(y_pred[y_test == k] == k).mean():.3f}")
    return net


if __name__ == "__main__":
    print(f"Gradient check, max relative error: {gradient_check():.2e}  (should be < 1e-5)")
    run("data/Clean_Dataset.csv")