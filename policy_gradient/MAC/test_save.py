import tensorflow as tf

with tf.Session() as sess:
	new_saver = tf.train.import_meta_graph('./log/policy_model_0.meta')
	new_saver.restore(sess, tf.train.latest_checkpoint('./log'))
	print(sess.run('w0:0').shape)
	print(sess.run('b0:0').shape)
	print(sess.run('w1:0').shape)
	print(sess.run('b1:0').shape)