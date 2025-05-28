import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
    
    def sigmoid(self, z):
        return 1/(1+ np.exp(-z))
    
    def fit(self, X, y):
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias=0
        
        for i in range(self.epochs):
            z= np.dot(X, self.weights) + self.bias
            predictions = self.sigmoid(z)
            
            # Compite cross-entropy loss
            if i%50 == 0:
                loss = -(1/num_samples) * np.sum(y*np.log(predictions) + (1-y) * np.log(1-predictions))
                print(f"Epoch: {i} Loss: {loss:.4f}")
                     
            #Compute gradients
            dw = (1/num_samples) * np.dot(X.T, (predictions-y))
            db = (1/num_samples) * np.sum(predictions - y)
            
            # Update weights and bias
            self.weights = self.weights - (self.learning_rate * dw)
            self.bias = self.bias - (self.learning_rate * db)          
    
    def predict(self, X, threshold=0.5):
        z = np.dot(X, self.weights) + self.bias
        predictions = self.sigmoid(z)
        predicted_classes = [1 if i > threshold else 0 for i in predictions]
        return np.array(predicted_classes)
    

if __name__ == "__main__":
    
    np.random.seed(42)
    
    X_class_0 = np.random.randn(100,2)*1.5 + np.array([1, 1])
    X_class_1 = np.random.randn(100,2)*1.5 + np.array([5, 5])
    X = np.vstack((X_class_0, X_class_1))
    y = np.array([0]*100 + [1]*100)
    print(X.shape, y.shape)
    print(X[:5], y[:5])
    print(X_class_0[:5], X_class_1[:5])
    
    logistic_regression = LogisticRegression(learning_rate = 0.01, epochs=500)
    logistic_regression.fit(X, y)
    logistic_regression.predict([[1, 1], [5, 5], [3, 3], [2, 2],[0.5,0.5]])