import numpy as np

INPUT_SHAPE = 18
HIDDEN_SIZE = 20
OUTPUT = 1
LR = 0.1
epochs = 2000

# Define sigmoid activation function
def sigmoid(x):
    return 1 / ( 1 + np.exp(-x))

# Define derivative of sigmoid function
def sigmoid_derivative(x):
    return x * (1 - x)

def ReLU(Z):
    return np.maximum(Z, 0)

def ReLU_deriv(Z):
    return Z > 0


def init_params(input_shape=18, hidden_size=20, output_size=1):
    W1 = np.random.rand(hidden_size, input_shape) 
    b1 = np.random.rand(hidden_size, 1) 
    W2 = np.random.rand(output_size, hidden_size) 
    b2 = np.random.rand(output_size, 1) 
    return W1, b1, W2, b2

def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = sigmoid(Z2)
    return Z1, A1, Z2, A2

def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    m = X.shape[1]
    dZ2 = A2 - Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2,axis=1)
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)
    
    return dW1, db1, dW2, db2
    

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1    
    W2 = W2 - alpha * dW2  
    b2 = b2 - alpha * db2    
    return W1, b1, W2, b2

def calculate_cost(A2, Y):
    m = len(Y)
    cost = -np.sum(np.multiply(Y, np.log(A2)) +  np.multiply(1-Y, np.log(1-A2))) /m
    cost = np.squeeze(cost)
    return cost

def get_accuracy(y_true, y_pred):
    correct = 0
    total = len(y_true)
    
    for i in range(total):
        if y_true[i] == y_pred[i]:
            correct += 1
    
    accuracy = correct / total
    
    return accuracy

def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    A2 = np.round(A2).squeeze()
    return A2


def gradient_descent(X, Y, alpha, iterations,input_shape, hidden_size, output_size):
    history = {"epoch":[],"loss":[],"accuracy":[]}
    W1, b1, W2, b2 = init_params(input_shape, hidden_size, output_size)
     
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)        
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        cost = calculate_cost(A2,Y)
        A2 = np.round(A2).squeeze()
        accuracy = get_accuracy(Y,A2)
        history['loss'].append(cost)
        history['epoch'].append(i)
        history['accuracy'].append(accuracy)
        if i % 100 == 0:
            print("Iteration: ", i)
            print("Cost is : ",cost)
            print("Accuracy is :",accuracy)
            print('################ \n')
    return W1, b1, W2, b2,history


W1, b1, W2, b2,history = gradient_descent(x_train, y_train, LR,epochs,
                          input_shape=INPUT_SHAPE, hidden_size=HIDDEN_SIZE, output_size=OUTPUT)

y_pred = make_predictions(x_test, W1, b1, W2, b2)
print("Model Accuracy on Test data :",round(get_accuracy(y_test,y_pred) * 100))