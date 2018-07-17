"""Here I will follow the toturial of tensorflow cookbook"""
"""It's good for operations on tensors and it has a lazy execution style"""
import tensorflow as tf
import numpy as np

# initialize twp constants
x1 = tf.constant([1,2,3,4])
x2 = tf.constant([5,6,7,8])

# lazy multiply
result = tf.multiply(x1, x2) # lazy eval. not actually doing anything
print(result)

# for interactive usage (python-way) we use the Session() method. This is the evaluation of the computational graph
sess = tf.Session()
print(sess.run(result))  # Note that the result is hadamard multiplication
sess.close()

# Another way of doing this:
with tf.Session() as sess:
    output = sess.run(result)
    print(output)


"""Starting from the beggining with the cookbook"""
# Hello World in tensorflow
message = tf.constant('Welcome to TensorFlow')
with tf.Session() as sess:
    print(sess.run(message).decode())

# Adding two vectors
v1 = tf.constant([1, 2, 3, 4])
v2 = tf.constant([2, 1, 5, 3])
v_add = tf.add(v1, v2)

with tf.Session()as sess:
    print(sess.run(v_add))

# or equivalently:
sess = tf.Session()
print(sess.run(v_add))
sess.close() # in this way, you need to close the session

# Cool thing the interactive session
sess = tf.InteractiveSession()
v1 = tf.constant([1, 2, 3, 4])
v2 = tf.constant([2, 1, 5, 3])

v_add = tf.add(v1, v2)
print(v_add.eval())  # this way you can see midterm results
sess.close()

"""
TensorFlow Data Objects:
1. Constants - values that cannot be changed
2. Variables - values that can be changed - like the weights that require updating every epoch
3. PlaceHolders - values that we feed to the comp. graph. for our porpuse these are the trainig examples.
                  we assign values to the placeholders one the graph is running.
"""

"""Basic Usage of the Constant Objects"""
t1 = tf.constant(4)
t2 = tf.constant([4, 3, 2])
zerof_t = tf.zeros([2, 3], tf.int32)
zeros_t2 = tf.zeros_like(t2)  # Can use constants to define a tensor
ones_t2 = tf.ones_like(t2)

# sequences:
linspace_t = tf.linspace(2.0, 5.0, 5)
range_t = tf.range(1, 10, 0.5)

# random tensors:
t_rand_normal = tf.random_normal(shape=[2, 3], mean=0.1, stddev=1.0)

# truncated normal - a normal distribution where each sample that is more than 2 std gets
# cropped and re-picked 
t_random = tf.truncated_normal([9, 9], stddev=1.0, seed=12)
# randomly crop a tensor - for example randomly cropping part of a picture

# constant only accepts numpy like arrays. NOT TENSORS!
tf_random_var = tf.constant(
    np.random.normal(loc=0.0, scale=1.0, size=(9, 9)).astype(np.float32)) 
t_cropped = tf.random_crop(tf_random_var, [2,2], seed=12)
#Sometimes we wish to randomize the training sample:
random_rows = tf.random_shuffle(t_random)
sess = tf.Session()
print(sess.run(tf_random_var))
print(sess.run(t_cropped)) # good we learned something
sess.close()

# for repeatness in experiemts you can do in the beggining:
tf.set_random_seed(54)

"""Basic Usage of the Variable Objects"""
rand_t = tf.random_uniform(shape=[50, 50], minval=0, maxval=10, seed=0)
t_a = tf.Variable(rand_t)
t_b = tf.Variable(rand_t)

# variables often represents the weights and biases of the network:
weights = tf.Variable(tf.random_normal(shape=[100, 100], mean=0.0, stddev=2.0))
bias = tf.Variable(tf.zeros([100], tf.float32), name='biases') # you can name a variables
# You can also initialize a Variable from an existing Variable
weights2 = tf.Variable(weights.initialized_value(), name='w2')
# We must explicitly decalre all the variables by declaring an Initializtion Operation Object:
initial_op = tf.global_variables_initializer()

# You can also initialize each variable by its own during the Sessins run:
bias = tf.Variable(tf.zeros([100, 100]))
with tf.Session() as sess:
    sess.run(bias.initializer)
# save Variables:
saver = tf.train.Saver() # a Saver object - not really sure why

# Very often we will use constant but big tensors. It's a good practice to save them as Variables:
large_array = np.zeros(2000)
t_large = tf.Variable(large_array, trainable=False)

"""Placeholders - Objects that feed data to the graph"""
x = tf.placeholder('float')
y = 2 * x
data = tf.random_uniform(shape=[4, 5], maxval=10)
with tf.Session() as sess:
    x_data = sess.run(data)  # Creating the data
    print(sess.run(y , feed_dict={x: x_data}))  # feeding the data inside the placeholder


"""Matrix Manipulation using TensorFlow"""

# This is for interactive session:
sess = tf.InteractiveSession()
I_matrix = tf.eye(5)  # 5x5 indentity matrix
print(I_matrix.eval())

# Same but with a Variable
X = tf.Variable(tf.eye(10))
X.initializer.run()  # remember we must initialize a Variable
print(X.eval())

# Create random matrix:
A = tf.Variable(tf.random_normal(shape=[5, 10]))
A.initializer.run()

# Multiply the two matrices above
product = tf.matmul(A, X)
print(product.eval())


# Create a random matrix of 1s and 0s, size 5x10
b = tf.Variable(tf.random_uniform([5, 10], minval=0, maxval=2, dtype=tf.int32))
b.initializer.run()
print(b.eval())
# Cast entries back to float. In operations on two or more matrices, all have same dtype
b_new = tf.cast(b, dtype=tf.float32)

# Add two matrices
t_sum = tf.add(product, b_new)
t_sub = tf.add(product, tf.negative(b_new))

print('Addition of Matrices: \n', t_sum.eval())
print('Subtraction of Matrices: \n', t_sub.eval())

# Create two random matrices
a = tf.Variable(tf.random_normal([4,5], stddev=2))
b = tf.Variable(tf.random_normal([4,5], stddev=2))


# Hadamard multiplication
A = a * b
# print(H.eval())

#Multiplication with a scalar 2
B = tf.scalar_mul(2, A)

# Element-wise division:
C = tf.div(a, b)
# print(C.eval())

# Element Wise remainder of division
D = tf.mod(a, b)
# print(D.eval())
init_op = tf.global_variables_initializer()

#Not sure why we need this writer thingie - maybe for tensorboard
with tf.Session() as sess:
     sess.run(init_op)
     writer = tf.summary.FileWriter('graphs', sess.graph)
     a,b,A_R, B_R, C_R, D_R = sess.run([a , b, A, B, C, D])
     print("a\n",a,"\nb\n",b, "a*b\n", A_R, "\n2*a*b\n", B_R, "\na/b\n", C_R, "\na%b\n", D_R)

writer.close()
"""Using tensorboard"""
""" In this folder type in the cmd: tensorboard --logdir=graphs and go to the URL provided
Really nice
"""
