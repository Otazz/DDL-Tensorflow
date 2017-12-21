# IBM DDL-Tensorflow
### Using DDL for distributed processing on GPU
#### Two examples:
- Simple MNIST Classifier using shallow neural network - ddl.py
- MNIST recognition using CNN - deep_mnist.py

####	Loading DDL library
	ddl = tf.load_op_library('/opt/DL/ddl-tensorflow/lib/ddl_MDR.so')

####	Initializing DDL with number of proccess and working directives with the CPU
	with tf.Session(config=config) as sess:
	  with tf.device('/cpu:0'):
	    rank, size, gpuid = sess.run(ddl.init(4, mode = '-mode r:4 -dump_iter 100'))

#### Using the gpu, start the graph definition inside this scope
	with tf.device('/gpu:%d' %gpuid):

#### Loading the Matrix Variables into the GPU's memory with ddl.bcast
	W = tf.Variable(ddl.bcast(tf.zeros([784, 10])))
  	b = tf.Variable(ddl.bcast(tf.zeros([10])))

#### After computing the gradients with Tensorflow's optmizer executing an all_reduce operation for the average and applying the result to the network
	opt = tf.train.GradientDescentOptimizer(0.5)
	
	grads_and_vars = opt.compute_gradients(cross_entropy)
  	grads, vars = zip(*grads_and_vars)
  	grads_and_vars_ddl = zip(ddl.all_reduce_n(grads, op='avg'), vars)
	
	objective = opt.apply_gradients(grads_and_vars_ddl)
	
#### Training the data in mini-batches dividing it in each GPU
	with tf.Session(config=config) as sess:
  	  sess.run(tf.global_variables_initializer())
  	    for i in range(1000):
	      batch_x, batch_y = mnist.train.next_batch(batch_size*size)
	      batch_x = np.split(batch_x,size)[rank]
	      batch_y = np.split(batch_y,size)[rank]
	      
	      sess.run(objective, feed_dict={x: batch_x, y_: batch_y})
