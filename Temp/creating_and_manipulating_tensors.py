import tensorflow as tf

'''

try:
	tf.contrib.eager.enable_eager_execution();
	print("TF imported with eager execution!");
except ValueError:
	print("TF already imported with eager execution!");

primes = tf.constant([2, 3, 5, 7, 11, 13], dtype = tf.int32);
print("primes:", primes);

ones = tf.ones([6], dtype = tf.int32);
print("ones:", ones);

justBeyondPrimes = tf.add(primes, ones);
print("just beyond primes:", justBeyondPrimes);

twos = tf.constant([2, 2, 2, 2, 2, 2], dtype = tf.int32);
primesDoubled = primes * twos;
print("primes doubled:", primesDoubled);

# 效果与先创建twos在相乘相同 
temp = primes * 2;
print("temp:", temp);
'''

'''
someMatrix = tf.constant([[1, 2, 3], [4, 5, 6]], dtype = tf.int32);
print(someMatrix);
print("value of someMatrix is:\n", someMatrix.numpy());

scalar = tf.zeros([]);
vector = tf.zeros([3]);

matrix = tf.zeros([2, 3]);

print("scalar has shape: ", scalar.get_shape(), " and value:\n", scalar.numpy());
print("vector has shape: ", vector.get_shape(), " and value:\n", vector.numpy());
print("matrix has shape: ", matrix.get_shape(), " and value:\n", matrix.numpy());

one = tf.constant(1, dtype = tf.int32);
print("one:", one);

justBeyondPrimes = tf.add(primes, one);
print("just beyond primes:", justBeyondPrimes);
'''
'''
import tensorflow as tf
primes = tf.constant([2, 3, 5, 7, 11, 13], dtype = tf.int32);
justUnderPrimesSquared = tf.pow(primes, 2) - 1;
print("just under primes squared:", justUnderPrimesSquared);

x = tf.constant([[5, 2, 4, 3], [5, 1, 6, -2], [-1, 3, -1, -2]], dtype = tf.int32);
y = tf.constant([[2, 3], [3, 5], [4, 5], [1, 6]], dtype = tf.int32);

matrixMultiplyResult = tf.matmul(x, y);
print(matrixMultiplyResult.numpy());

matrix = tf.constant([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]], dtype = tf.int32);

reshapedMatrix_2x8 = tf.reshape(matrix, [2, 8]);
reshapedMatrix_4x4 = tf.reshape(matrix, [4, 4]);

print(matrix.numpy());
print(reshapedMatrix_2x8);
print(reshapedMatrix_4x4);

reshapedMatrix_2x2x4 = tf.reshape(matrix, [2, 2, 4]);
oneDimensionalVector = tf.reshape(matrix, [16]);

print(reshapedMatrix_2x2x4.numpy());
print(oneDimensionalVector.numpy());

'''

'''
import tensorflow as tf

a = tf.constant([5, 3, 2, 7, 1, 4]);
b = tf.constant([4, 6, 3]);
reshapedA_2x3 = tf.reshape(a, [2, 3]);
reshapedB_3x1 = tf.reshape(b, [3, 1]);

c = tf.matmul(reshapedA_2x3, reshapedB_3x1);
print(c.numpy());
'''
try:
	tf.contrib.eager.enable_eager_execution();
	print("TF imported with eager execution!");
except:
	print("TF already imported with eager execution!");

# Create a scalar variable with the initial value 3.
v = tf.contrib.eager.Variable([3])

# Create a vector variable of shape [1, 4], with random initial values,
# sampled from a normal distribution with mean 1 and standard deviation 0.35.
w = tf.contrib.eager.Variable(tf.random_normal([1, 4], mean=1.0, stddev=0.35))

print("v:", v.numpy())
print("w:", w.numpy())

v = tf.contrib.eager.Variable([3])
print(v.numpy())

tf.assign(v, [7])
print(v.numpy())

v.assign([5])
print(v.numpy())

v = tf.contrib.eager.Variable([[1, 2, 3], [4, 5, 6]])
print(v.numpy())

try:
	print("Assigning [7, 8, 9] to v")
	v.assign([7, 8, 9])
except ValueError as e:
  print("Exception:", e)

import tensorflow as tf

try:
	tf.contrib.eager.enable_eager_execution();

	print("TF imported with eager execution!");
except:
	print("TF already imported with eager execution!");

die1 = tf.contrib.eager.Variable(tf.random_uniform([10, 1], minval = 1, maxval = 7, dtype = tf.int32));

die2 = tf.contrib.eager.Variable(tf.random_uniform([10, 1], minval = 1, maxval = 7, dtype = tf.int32));

sum = tf.add(die2, die2);

resultMatrix = tf.concat(values = [die1, die2, sum], axis = 1);
print(resultMatrix.numpy());


die1 = tf.contrib.eager.Variable(
    tf.random_uniform([10, 1], minval=1, maxval=7, dtype=tf.int32))
die2 = tf.contrib.eager.Variable(
    tf.random_uniform([10, 1], minval=1, maxval=7, dtype=tf.int32))

dice_sum = tf.add(die1, die2)
resulting_matrix = tf.concat(values=[die1, die2, dice_sum], axis=1)


print(resulting_matrix.numpy());