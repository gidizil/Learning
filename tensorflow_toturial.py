"""Here I will follow the toturial of tensorflow cookbook"""
"""It's good for operations on tensors and it has a lazy execution style"""
import tensorflow as tf


#intialize twp constants
x1 = tf.constant([1,2,3,4])
x2 = tf.constant([5,6,7,8])

#lazy multiply
result = tf.multiply(x1, x2) #lazy eval. not actually doing anything
print(result)

#for interactive usage (python-way) we use the Session() method. This is the evaluation of the computational graph
sess = tf.Session()
print(sess.run(result)) #Note that the result is hadamard multiplication
sess.close()

#Another way of doing this:
with tf.Session() as sess:
    output = sess.run(result)
    print(output)


"""Starting from the beggining with the cookbook"""
#Hello World in tensorflow
message = tf.constant('Welcome to TensorFlow')
with tf.Session() as sess:
    print(sess.run(message).decode())

# Adding two vectors
v1 = tf.constant([1, 2, 3, 4])
v2 = tf.constant([2, 1, 5, 3])
v_add = tf.add(v1, v2)

with tf.Session()as sess:
    print(sess.run(v_add))

#or equivalently:
sess = tf.Session()
print(sess.run(v_add))
sess.close() # in this way, you need to close the session

#Cool thing the interactive session
sess = tf.InteractiveSession()
v1 = tf.constant([1, 2, 3, 4])
v2 = tf.constant([2, 1, 5, 3])

v_add = tf.add(v1, v2)
print(v_add.eval()) #this way you can see midterm results
sess.close()

"""
TensorFlow Data Objects:
1. Constants - values that cannot be changed
2. Variables - values that can be changed - like the weights that require updating every epoch
3. PlaceHolders - values that we feed to the comp. graph. for our porpuse these are the trainig examples.
                  we assign values to the placeholders one the graph is running.
"""
