import tensorflow as tf

node1 = tf.constant(3.0, dtype=tf.float32)
node2 = tf.constant(4.0) #also tf.float32 implicitly
#print(node1, node2) 
#Tensor("Const_1:0", shape=(), dtype=float32) Tensor("Const_2:0", shape=(), dtype=float32)
sess = tf.Session()
# To actually evaluate the nodes, we must run the computational graph within a session. 
# A session encapsulates the control and state of the TensorFlow runtime.

#print(sess.run([node1, node2])) #[3.0, 4.0]
node3 = tf.add(node1, node2)
#print("node3: ", node3) #node3:  Tensor("Add:0", shape=(), dtype=float32)
#print("sess.run(node3): ", sess.run(node3)) #sess.run(node3):  7.0

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a+b # + provides a shortcut for tf.add(a,b)
print(sess.run(adder_node, {a:3,b:4.5})) # 7.5
print(sess.run(adder_node, {a:[1,3],b:[2,4]})) # [ 3.  7.]

add_and_triple = adder_node * 3.
print(sess.run(add_and_triple, {a:3, b:4.5})) # 22.5

#To make the model trainable, 
#we need to be able to modify the graph to get new outputs with the same input. 
#Variables allow us to add trainable parameters to a graph. 
#They are constructed with a type and initial value:
W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W * x + b
#Constants are initialized when you call tf.constant and their value can never change. 
#By contrast, variables are not initialized when you call tf.Variable. 
#To initialize all the variables in a TensorFlow program, 
#you must explicitly call a special operation as follows:
init = tf.global_variables_initializer()
sess.run(init)
print(sess.run(linear_model, {x:[1,2,3,4]})) # [ 0.  0.30000001  0.60000002  0.90000004]

#We've created a model, but we don't know how good it is yet.
# To evaluate the model on training data,
# we need a y placeholder to provide the desired values,
# and we need to write a loss function.
y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)
# loss function measures how far apart the current model is from the provided data.
loss = tf.reduce_sum(squared_deltas) 
print(sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]})) #producing the loss value 23.66

"""
We could improve this 'manually' by reassigning the values of W and b 
to the perfect values of -1 and 1. 
A variable is initialized to the value provided to tf.Variable but 
can be changed using operations like tf.assign.
For example, W=-1 and b=1 are the optimal parameters for our model. 
We can change W and b accordingly:
"""
fixW = tf.assign(W, [-1.])
fixb = tf.assign(b, [1.])
sess.run([fixW, fixb]) # [array([-1.], dtype=float32), array([ 1.], dtype=float32)]
print(sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]})) # 0.0
# We guessed the "perfect" values of W and b, 
# but the whole point of machine learning is to find the correct model parameters automatically. 
# We will show how to accomplish this in the next section.


"""
A complete discussion of machine learning is out of the scope of this tutorial. 
However, TensorFlow provides optimizers that slowly change each variable in order to 
minimize the loss function. 

The simplest optimizer is gradient descent. 
It modifies each variable according to the magnitude of the derivative of loss 
with respect to that variable. 

In general, computing symbolic derivatives manually is tedious and error-prone. 
Consequently, TensorFlow can automatically produce derivatives given only a description of 
the model using the function tf.gradients. 

For simplicity, optimizers typically do this for you. For example,
"""
optimizer = tf.train.GradientDescentOptimizer(0.01) # optimizer
train = optimizer.minimize(loss)
sess.run(init) #reset values to incorrect defaults.
# training data
x_train = [1,2,3,4]
y_train = [0,-1,-2,-3]
for i in range(1000):
  sess.run(train, {x:x_train,  y:y_train})

# evaluate training accuracy
curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x:x_train, y:y_train})
print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))
#W: [-0.9999969] b: [ 0.99999082] loss: 5.69997e-11
