"""This time the excerise will be on logistic regression Gradient Descent"""
import numpy as np
import matplotlib.pyplot as plt


def gen_sample(instances, features):
    """Generate the X, y sample and the true weights"""
    X = np.random.normal(loc=0.0, scale=4.0, size=[instances, features])
    bias = np.ones(shape=[instances, 1])
    X = np.column_stack((bias, X))
    true_weights = np.random.chisquare(5, size=[features + 1, 1])
    y_val = X.dot(true_weights)
    y = np.where(y_val > 0, 1, 0)
    return X, y, true_weights

def gen_sample_2(instances, features):
    X1 = np.random.multivariate_normal(5*np.ones(features), np.eye(features), size=instances) + 1.15
    X2 = np.random.multivariate_normal(3*np.ones(features), np.eye(features), size=instances)
    bias = np.ones(shape=[2 * instances, 1])
    y1 = np.ones([instances, 1])
    y2 = np.zeros([instances, 1])
    X = np.vstack((X1, X2))
    X = np.column_stack((bias, X))
    y = np.vstack((y1, y2))
    return X, y
def gen_intial_weights(features):
    """Generate initial weights"""
    w = np.random.normal(loc=0.0, scale=0.5, size=[features + 1, 1])
    return w

def calc_sigmoid(z):
    """Calculate sigmoid function given dataset and weight"""
    return 1 / (1 + np.exp(-z))

def calc_sigmoid_vec(X, w):
    sigmoid = lambda z: 1 / (1 + np.exp(-z))
    sigmoid = np.vectorize(sigmoid)
    sigmoid_vec = sigmoid(X.dot(w))
    return sigmoid_vec

def calc_gradient_step(X, y, w, gamma, sigmoid_vec):
    error_vec = sigmoid_vec - y
    return gamma * X.transpose().dot(error_vec)



def make_gradient_step(current_w, gd_step):
    w = current_w - gd_step
    return w

def calc_error_rate(sigmoid_vec, y):
    sigmoid_est = np.where(sigmoid_vec >= 0.5, 1, 0)
    errors = np.where((sigmoid_est - y) != 0, 1, 0)
    error_rate = np.sum(errors) / len(y)
    return error_rate

def plot_classifier(w):
    intercept = - w[0] / w[2]
    slope = - w[1] / w[2]
    line = lambda x: intercept + slope * x
    x = np.arange(0, 8, 0.2)
    line = np.vectorize(line)
    y = line(x)
    plt.plot(x, y)


"""Now lets run our experiment"""
instances = 15000
features = 100
gamma = 0.0001
min_error_rate = 0.015
counter = 0
error_rate_vec = []

# X, y, true_weights = gen_sample(instances, features)
# w = gen_intial_weights(features)
# sigmoid_vec = calc_sigmoid_vec(X, w)
# error_rate = calc_error_rate(sigmoid_vec, y)
# while error_rate > min_error_rate:
#     sigmoid_vec = calc_sigmoid_vec(X, w)
#     gd_step = calc_gradient_step(X, y, w, gamma, sigmoid_vec)
#     w = make_gradient_step(w, gd_step)
#     error_rate = calc_error_rate(sigmoid_vec, y)
#     error_rate_vec.append(error_rate)
#     #print(error_rate)
#     counter += 1
# print(counter)
# print('True Weights:', true_weights)
# print('Final Weights:', w)
# plt.plot(error_rate_vec)
# plt.show()

# TODO: generate sample X, y using two clusters where linear shattering of the data is not possible
instances = 200
features = 2
X, y = gen_sample_2(instances, features)
x1, y1 = X[0:instances, 1], X[0:instances, 2]
x2, y2 = X[instances::, 1], X[instances::, 2]

# plt.show()

#Second Experiment using the second method of sampling:
X,  y  = gen_sample_2(instances, features)
w = gen_intial_weights(features)
sigmoid_vec = calc_sigmoid_vec(X, w)
error_rate = calc_error_rate(sigmoid_vec, y)
error_rate_vec = []
while error_rate > min_error_rate:
    sigmoid_vec = calc_sigmoid_vec(X, w)
    gd_step = calc_gradient_step(X, y, w, gamma, sigmoid_vec)
    w = make_gradient_step(w, gd_step)
    error_rate = calc_error_rate(sigmoid_vec, y)
    error_rate_vec.append(error_rate)

plt.plot(error_rate_vec)
plt.show()


plt.scatter(x1, y1, color='blue')
plt.scatter(x2, y2, color='red')
plot_classifier(w)
plt.show()



