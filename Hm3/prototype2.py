import pickle
import os
import pandas as pd
import numpy as np

train_file = "extended_mnist_train.pkl"
test_file = "extended_mnist_test.pkl"

with open(train_file, "rb") as fp:
    train = pickle.load(fp)

with open(test_file, "rb") as fp:
    test = pickle.load(fp)

train_data = []
train_labels = []
for image, label in train:
    train_data.append(image.flatten()) #ex. train_data[0] will contain all pixels values in the first image
    train_labels.append(label)

#prediction only - so no labels in test
test_data = []
for image, label in test:
    test_data.append(image.flatten())

X_train = np.vstack(train_data).astype(np.float32) / 255.0
y_train = np.array(train_labels, dtype=np.int64)
X_test  = np.vstack(test_data).astype(np.float32) / 255.0

N, D = X_train.shape
K = int(y_train.max() + 1)

print(X_train.shape, X_train.dtype, X_train.min(), X_train.max())
print(y_train.shape, y_train.dtype, np.unique(y_train))
print(X_test.shape)

class MLP:
    def __init__(self, n_features=784, n_hidden=100, n_classes=10, lambda_l2=1e-4, lr=0.1, seed=0):
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.lambda_l2 = lambda_l2
        self.lr = lr

        self.rng = np.random.default_rng(seed)

        #he initialization for ReLU
        self.W1 = self.rng.normal(
            0.0, np.sqrt(2.0 / n_features), size=(n_features, n_hidden)
        ).astype(np.float32)
        self.b1 = np.zeros(n_hidden, dtype=np.float32)

        self.W2 = self.rng.normal(
            0.0, np.sqrt(2.0 / n_hidden), size=(n_hidden, n_classes)
        ).astype(np.float32)
        self.b2 = np.zeros(n_classes, dtype=np.float32)

    def relu(self, Z):
        return np.maximum(Z, 0.0)

    def softmax(self, Z):
        Zs = Z - Z.max(axis=1, keepdims=True)
        expZ = np.exp(Zs, dtype=np.float32)
        return expZ / expZ.sum(axis=1, keepdims=True)

    #forward pass that returns all intermediates (useful for debugging)
    def forward(self, X):
        Z1 = X @ self.W1 + self.b1 #(N, 100)
        A1 = self.relu(Z1) #(N, 100)
        Z2 = A1 @ self.W2 + self.b2 #(N, 10)
        P  = self.softmax(Z2) #(N, 10)
        return Z1, A1, Z2, P

    #for prediction we only need output logits
    def logits(self, X):
        Z1 = X @ self.W1 + self.b1
        A1 = self.relu(Z1)
        Z2 = A1 @ self.W2 + self.b2
        return Z2

    def ce_with_l2_from_probs(self, P, y):
        p_true = P[np.arange(P.shape[0]), y]
        ce = -np.mean(np.log(p_true + 1e-12))

        #L2 regularization on both W1 and W2
        reg = 0.5 * self.lambda_l2 * (np.sum(self.W1 * self.W1) + np.sum(self.W2 * self.W2))
        return ce + reg
    
    #one gradient descent step using backpropagation
    def gradient_descent(self, X, y):
        N = X.shape[0]

        Z1 = X @ self.W1 + self.b1
        A1 = self.relu(Z1)
        Z2 = A1 @ self.W2 + self.b2
        P  = self.softmax(Z2)

        #gradient at output: dL/d(logits) = (P - Y_one_hot)/N
        P[np.arange(N), y] -= 1.0
        P /= N # now P is G2 = dL/dZ2

        G2 = P
        dW2 = A1.T @ G2 + self.lambda_l2 * self.W2   # (100, 10)
        db2 = G2.sum(axis=0)  # (10,)

        #backprop into hidden: G1 = G2 @ W2^T, then ReLU derivative
        G1 = G2 @ self.W2.T # (N, 100)
        G1[Z1 <= 0] = 0.0 # ReLU'(Z1)

        dW1 = X.T @ G1 + self.lambda_l2 * self.W1 # (784, 100)
        db1 = G1.sum(axis=0) # (100,)

        #-gradient step
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    def predict(self, X):
        return self.logits(X).argmax(axis=1)

    def accuracy(self, X, y):
        return (self.predict(X) == y).mean()

    def fit(self, X, y, X_val=None, y_val=None, epochs=90, batch_size=256, shuffle=True, print_every=10, seed=0):

        best_W1 = self.W1.copy()
        best_b1 = self.b1.copy()
        best_W2 = self.W2.copy()
        best_b2 = self.b2.copy()

        best_score = 0.0 #train acc

        rng = np.random.default_rng(seed)
        N = X.shape[0]

        for ep in range(1, epochs + 1):
            idx = np.arange(N)
            if shuffle:
                rng.shuffle(idx)

            #mini-batches for gradient desc
            for s in range(0, N, batch_size):
                ii = idx[s:s + batch_size]
                self.gradient_descent(X[ii], y[ii])

            #logging
            if ep % print_every == 0:
                _, _, _, P_tr = self.forward(X)
                train_loss = self.ce_with_l2_from_probs(P_tr, y)
                train_acc = self.accuracy(X, y)

                if X_val is not None:
                    _, _, _, P_val = self.forward(X_val)
                    val_loss = self.ce_with_l2_from_probs(P_val, y_val)
                    val_acc  = self.accuracy(X_val, y_val)
                    print(
                          f"epoch {ep:03d} | "
                          f"train loss={train_loss:.4f} acc={train_acc:.4f} | "
                          f"val loss={val_loss:.4f} acc={val_acc:.4f}"
                    )
                    score = val_acc
                else:
                    print(
                        f"epoch {ep:03d} | "
                        f"train loss={train_loss:.4f} acc={train_acc:.4f}"
                    )
                    score = train_acc

                #save best parameters
                if score > best_score:
                    best_score = score
                    best_W1 = self.W1.copy()
                    best_b1 = self.b1.copy()
                    best_W2 = self.W2.copy()
                    best_b2 = self.b2.copy()

        #restore best model
        self.W1, self.b1 = best_W1, best_b1
        self.W2, self.b2 = best_W2, best_b2


model = MLP(n_features=784, n_hidden=100, n_classes=10, lambda_l2=1e-4, lr=0.1, seed=42)

#train on train subset
model.fit(X_train, y_train, epochs=240, batch_size=256, shuffle=True, print_every=10)

print("Final train accuracy:", model.accuracy(X_train, y_train))
#print("Final val accuracy:",   model.accuracy(X_val, y_val))

#fr tst
predictions = model.predict(X_test).astype(int) #extracting for each sample the label with the highest logit
submission = pd.DataFrame({
    "ID": np.arange(len(predictions), dtype=int),
    "target": predictions
})
submission.to_csv("submission_mlp.csv", index=False)
print("Saved submission_mlp.csv")