# -*- coding:utf-8 -*-

import numpy as np
import tensorflow as tf
from tensorflow.python.data import Dataset

a = np.array(range(0, 10));
print(a);

dataset = Dataset.from_tensor_slices(a);

dataset = dataset.repeat(None);

dataset = dataset.batch(4);


iter = dataset.make_one_shot_iterator();
el = iter.get_next();

with tf.Session() as sess:
	print(sess.run(el));
	print(sess.run(el));
	print(sess.run(el));
	print(sess.run(el));
	print(sess.run(el));
	print(sess.run(el));
	print(sess.run(el));
	print(sess.run(el));