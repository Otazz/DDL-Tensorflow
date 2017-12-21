from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = None

#Inicializando o ddl carregando a biblioteca
ddl = tf.load_op_library('/opt/DL/ddl-tensorflow/lib/ddl_MDR.so')

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

#Utilizando o ddl.init para pegar as informacoes relevantes a GPU e procesos
with tf.Session(config=config) as sess:
    with tf.device('/cpu:0'):
        rank, size, gpuid = sess.run(ddl.init(4, mode = '-mode r:4 -dump_iter 100'))


#Utilizando GPU para fazer as operacoes
with tf.device('/gpu:%d' %gpuid):
  batch_size = 100
  #Importando o dataset
  mnist = input_data.read_data_sets('/tmp/data', one_hot=True)

  #Criando o modelo, ddl.bcast carrega os parametros para a memoria da gpu
  x = tf.placeholder(tf.float32, [None, 784])
  W = tf.Variable(ddl.bcast(tf.zeros([784, 10])))
  b = tf.Variable(ddl.bcast(tf.zeros([10])))
  y = tf.matmul(x, W) + b

  #Definindo a funcao de perda
  y_ = tf.placeholder(tf.float32, [None, 10])
  cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

  #Optimizador
  opt = tf.train.GradientDescentOptimizer(0.5)

  #Computando os gradientes da rede (lista de pares (gradiente,variavel))
  grads_and_vars = opt.compute_gradients(cross_entropy)
  grads, vars = zip(*grads_and_vars)

  #Utilizando a funcao all_reduce_n para calcular as medias dos gradientes de cada variavel
  #juntando os valores de cada GPU
  grads_and_vars_ddl = zip(ddl.all_reduce_n(grads, op='avg'), vars)

  #Aplicando os gradientes a rede
  objective = opt.apply_gradients(grads_and_vars_ddl)

  #Testar os valores e gerar a acuracia do modelo
  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session(config=config) as sess:
  sess.run(tf.global_variables_initializer())
  # Treinamento
  for i in range(1000):
    batch_x, batch_y = mnist.train.next_batch(batch_size*size)
    batch_x = np.split(batch_x,size)[rank]
    batch_y = np.split(batch_y,size)[rank]

    sess.run(objective, feed_dict={x: batch_x, y_: batch_y})
    if not i % 100:
      loss, acc = sess.run([cross_entropy, accuracy], feed_dict={x: batch_x,
                                                              y_: batch_y})
      print("MPI "+str(rank)+"] Iter " + str(i) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))


  #Teste
  print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                        y_: mnist.test.labels}))