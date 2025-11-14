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

#prediction only - so no labels
test_data = []
for image, label in test:
    test_data.append(image.flatten())

X_train = np.vstack(train_data).astype(np.float32) / 255.0   #build a n x 784 matrix (a row with 784 elems for eacch image from train data)
y_train = np.array(train_labels, dtype=np.int64)  #creating a numpy array of labels
X_test  = np.vstack(test_data).astype(np.float32) / 255.0    #build a m x 784 matrix (a row with 784 elems for each image from test data)

N, D = X_train.shape #matrix coords
K = int(y_train.max() + 1) #nr of labels in train

print(X_train.shape, X_train.dtype, X_train.min(), X_train.max())
print(y_train.shape, y_train.dtype, np.unique(y_train))
print(X_test.shape)

class Perceptron:
    def __init__(self, n_features = 784, n_classes = 10, lambda_l2 = 1e-4, lr = 0.1):
        self.n_features = n_features
        self.n_classes = n_classes
        self.lambda_l2 = lambda_l2
        self.lr = lr

        #initialize both(matrix with weights(10 x 784) and the one with biases - one for each perceptron/class) with elems eq to 0
        self.W = np.zeros((self.n_features, self.n_classes), dtype=np.float32)
        self.b = np.zeros(self.n_classes, dtype=np.float32)

    #multiply samples x features(pixel color) with feature x classes(each column stores the weights for each of the 784 pixels)
    #we get on each row in the result the sum of pixel*weights for each class
    #then we add bias
    def logits(self, X: np.ndarray) -> np.ndarray:
        return X @ self.W + self.b
    
    def softmax(self, Z):
        Zs = Z - Z.max(axis=1, keepdims=True)
        expZ = np.exp(Zs)
        return expZ / expZ.sum(axis=1, keepdims=True) 
    
    def ce_with_l2_from_probs(self, P, y):
        #P the matrix with softmax values, y tha meatrix with the correcct label for each sample
        #returns mean cross-entropy + L2 penalty

        # mean cross-entropy
        p_true = P[np.arange(P.shape[0]), y]  #p_true is vector(1 x N) with true-class probs for each sample
        ce = -np.mean(np.log(p_true + 1e-12))  #average over samples

        #L2 regularization on W
        reg = 0.5 * self.lambda_l2 * np.sum(self.W * self.W)

        return ce + reg
    
    def gradient_descent(self, X, y):
        N = X.shape[0]

        #logits -> softmax
        Z = self.logits(X)
        P = self.softmax(Z)

        #G = (P - Y) / N
        P[np.arange(N), y] -= 1.0
        P /= N   #P the new G

        #gradients
        dW = X.T @ P + self.lambda_l2 * self.W
        db = P.sum(axis=0)

        self.W -= self.lr * dW
        self.b -= self.lr * db

    def predict(self, X):
        return self.logits(X).argmax(axis=1)

    def accuracy(self, X, y):
        return (self.predict(X) == y).mean()
    
    def fit(self, X, y, epochs=50, batch_size=256, shuffle=True, print_every=10, seed=0):
        best_W = self.W.copy()
        best_b = self.b.copy()
        best_accuracy = 0
        rng = np.random.default_rng(seed)
        N = X.shape[0]

        for ep in range(1, epochs + 1):
            idx = np.arange(N)
            if shuffle:
                rng.shuffle(idx) #mixing up idx for generalisation

            for s in range(0, N, batch_size):
                ii = idx[s:s+batch_size]  #ii contains the indices in one slice
                self.gradient_descent(X[ii], y[ii])

            if ep % print_every == 0:
                logits = self.logits(X)
                P = self.softmax(logits)
                loss = self.ce_with_l2_from_probs(P, y)
                acc = self.accuracy(X, y)
                print(f"epoch {ep:03d} | loss={loss:.4f} | accuracy={acc:.4f}")

                if acc > best_accuracy:
                    best_accuracy = acc
                    best_b = self.b.copy()
                    best_W = self.W.copy()

        self.b = best_b
        self.W = best_W


#withGalaxy
model = Perceptron(n_features=784, n_classes=10, lambda_l2=1e-5, lr=0.1)
model.fit(X_train, y_train, epochs=190, batch_size=256, shuffle=True, print_every=10)

print("Train accuracy:", model.accuracy(X_train, y_train))

#fr tst
predictions = model.predict(X_test).astype(int)  #extracting for each sample the label with the highest logit
submission = pd.DataFrame({
    "ID": np.arange(len(predictions), dtype=int),
    "target": predictions
})
submission.to_csv("submission.csv", index=False)
print("Saved submission.csv")