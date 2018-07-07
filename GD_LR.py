"""
Mission here is to create Gradient Descent Linear regression and
Stochastic Gradient Descnet.
"""

# First thing - Basic Gradient descent of a function:
#Example function f(x) = 3x^4 + 2x^2 +4x; f'(x) = 12x^3 + 4x + 4

d_f = lambda x: 12*x**3 + 4*x + 4  # The derivative
gamma = 0.01  # Step size
eps = 0.00001 # Ttolerance to local minima
x_diff = 1  # Intial value of differences between steps. initialization must be bigger than eps
curr_x = 1  # Initial value to start descending from
#Optionl add maximum iteration limit
iteration_counter = 0
while x_diff > eps:
    prev_x = curr_x
    curr_x -= gamma * d_f(curr_x)
    x_diff = abs(curr_x - prev_x)
    iteration_counter += 1
print('Local minima at x =', curr_x)
print('Number of iterations', iteration_counter)


"""Now For Least Squares Linear Regression using gradient descent"""
import numpy as np
import matplotlib.pyplot as plt
def gen_sample():
    """Return a data set X, and target variable y. Including intercept"""
    X = np.random.normal(loc=0.0, scale=5.0, size=[1000, 10])
    intercept = np.array(np.ones(shape=[1000, 1]))
    X = np.column_stack((intercept, X))
    true_weights = np.random.chisquare(5, size=[11, 1])
    y = X.dot(true_weights)
    y += np.random.normal(loc=0.0, scale=0.2,size=[1000, 1])
    return X, y, true_weights

def gen_initial_weights():
    """Generate the initial weights to start descending from"""
    return np.random.normal(loc=0.0, scale=0.5, size=[11, 1])
def calc_gradient_step(gamma, y, X, w):
    """Calculates the gradient based on the current position"""
    diff_vec = (y - X.dot(w)).transpose()
    grad_step = gamma * diff_vec.dot(X)
    return grad_step

def make_gd_step(current_w,grad_step):
    """Actually make the GD step"""
    w = current_w + grad_step.transpose()
    return w

def calc_sse(X, y, w):
    """Calculate the SSE to see the behavior"""
    sse = np.linalg.norm(y - X.dot(w))
    return sse

#Lets test it:
X, y, true_weights = gen_sample()
w = gen_initial_weights()
gamma = 0.00001
precision = 0.1
iterations = 0
print('true weights:', true_weights)
sse_vec = []
sse_vec.append(calc_sse(X, y, w))
while iterations < 20 :  # One can use precision to stop iterating if SSE is small enough
    sse_vec.append(calc_sse(X, y, w))
    gd_step = calc_gradient_step(gamma, y, X, w)
    w = make_gd_step(w, gd_step)
    iterations += 1
#print('SSE: ', sse)
print('final_weights:', w)
plt.plot(sse_vec)
plt.show()


    
    


