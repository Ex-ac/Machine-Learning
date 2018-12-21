import tensorflow as tf
import matplotlib.pyplot as plt
import  pandas as pd

g = tf.Graph();

with g.as_default():
    x = tf.constant(8, name = "xConst");
    y = tf.constant(5, name = "yConst");
    z = tf.constant(4, name = "zConst");
    temp = tf.add(x, y, name = "temp");
    sum = tf.add(temp, z, name = "sum");

    with tf.Session() as sess:
        print(sum.eval());