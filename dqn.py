import tensorflow as tf
import database as db
import numpy as np
import time
import os
import logger

DQN_TRAINING = True
TEACHING = False

class DQN():

	def __init__(self,
	             random_pick_p_start = 1,
	             random_pick_p_end = 0.05,
	             random_pick_peiriod = 30000,
	             batch_size = 4,
	             image_size = 84,
	             input_channels = 4,
	             minibatch_size=16,
	             discount_rate=0.95,
	             first_conv_stride = 4,
	             first_conv_patch_size = 8,
	             first_conv_out_depth = 16,
	             second_conv_stride = 2,
	             second_conv_patch_size = 4,
	             second_conv_out_depth = 32,
	             first_fully_connect_layer_size = 256,
	             second_fully_connect_layer_size = 256,
	             single_state_size = 1,
	             output = 5,  # up down left right fire
					target_network_update_rate=0.01
	             ):

		self.graph = tf.get_default_graph()
		self.s = tf.Session(graph=self.graph)
		#self.optimizer = tf.train.RMSPropOptimizer(learning_rate= 0.001, decay=0.9)

		self.batch_size = batch_size
		self.image_size = image_size
		self.input_channels = input_channels
		self.minibatch_size = minibatch_size
		self.discount_rate = discount_rate
		self.q_result = np.array(np.zeros((self.minibatch_size,1)))
		self.first_conv_stride = first_conv_stride
		self.first_conv_patch_size = first_conv_patch_size
		self.first_conv_out_depth = first_conv_out_depth
		self.second_conv_stride = second_conv_stride
		self.second_conv_patch_size = second_conv_patch_size
		self.second_conv_out_depth = second_conv_out_depth
		self.first_fully_connect_layer_size = first_fully_connect_layer_size
		self.second_fully_connect_layer_size = second_fully_connect_layer_size
		self.output = output
		self.single_state_size = single_state_size
		self.random_pick_p_start = random_pick_p_start
		self.random_pick_p_end = random_pick_p_end
		self.random_action_porb = random_pick_p_start
		self.random_pick_peiriod = random_pick_peiriod

		self.variable_define()

		self.error, self.optimizer_op = self.training_computation()

		'''
		self.saver = tf.train.Saver(var_list={'l1_w': self.layer1_weights,
		                            'l1_b': self.layer1_biases,
		                            'l2_w': self.layer2_weights,
		                            'l2_b': self.layer2_biases,
		                            'l3_w': self.layer3_weights,
		                            'l3_b': self.layer3_biases,
		                            'l4_w': self.layer4_weights,
		                            'l4_b': self.layer4_biases},
		                            max_to_keep=2)
		'''

		self.saver = tf.train.Saver()


		self.s.run(tf.initialize_all_variables())


	def variable_define(self):

		#input data
		self.actions = tf.placeholder(tf.int8, shape=(self.minibatch_size, 1))
		self.rewards = tf.placeholder(tf.float32, shape=(self.minibatch_size, 1))

		self.s0 = tf.placeholder(tf.float32, shape=(
			self.minibatch_size, self.image_size, self.image_size, self.input_channels))

		self.s1 = tf.placeholder(tf.float32, shape=(
			self.minibatch_size, self.image_size, self.image_size, self.input_channels))

		self.current_screen = tf.placeholder(tf.float32, shape=(
			self.single_state_size, self.image_size, self.image_size, self.input_channels))

		self.layer1_weights = tf.Variable(tf.truncated_normal(
          [self.first_conv_patch_size, self.first_conv_patch_size, self.input_channels, self.first_conv_out_depth],
			stddev=0.1), name='l1_w')

		self.layer1_biases = tf.Variable(tf.zeros([self.first_conv_out_depth]), name='l1_b')

		self.layer2_weights = tf.Variable(tf.truncated_normal(
          [self.second_conv_patch_size, self.second_conv_patch_size, self.first_conv_out_depth, self.second_conv_out_depth],
			stddev=0.1), name='l2_w')
		self.layer2_biases = tf.Variable(tf.constant(1.0, shape=[self.second_conv_out_depth]), name='l2_b')

		layer3_input_size = 9 * 9 * self.second_conv_out_depth
		self.layer3_weights = tf.Variable(tf.truncated_normal(
          [layer3_input_size, self.first_fully_connect_layer_size], stddev=0.1), name='l3_w')
		self.layer3_biases = tf.Variable(tf.constant(1.0, shape=[self.first_fully_connect_layer_size]), name='l3_b')

		self.layer4_weights = tf.Variable(tf.truncated_normal(
          [self.first_fully_connect_layer_size, self.output], stddev=0.1), name='l4_w')
		self.layer4_biases = tf.Variable(tf.constant(1.0, shape=[self.output]), name='l4_b')

		#self.training_computation()


		self.q_real = tf.placeholder(tf.float32, shape=(self.minibatch_size, 1))

	def model(self, data):

		#data = tf.nn.l2_normalize(data, 0, epsilon=1e-12, name=None)

		#data = tf.nn.l2_normalize(data, 1, epsilon=1e-12, name=None)

		self.normalized_data = data

		conv = tf.nn.conv2d(data, self.layer1_weights,
							[1, self.first_conv_stride, self.first_conv_stride, 1],
							padding='VALID')
		hidden = tf.nn.relu(conv + self.layer1_biases)

		conv = tf.nn.conv2d(hidden, self.layer2_weights, [1, self.second_conv_stride,self.second_conv_stride, 1],
							padding='VALID')

		hidden = tf.nn.relu(conv + self.layer2_biases)
		shape = hidden.get_shape().as_list()

		reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])

		hidden = tf.nn.relu(tf.matmul(reshape, self.layer3_weights) + self.layer3_biases)

		output = tf.matmul(hidden, self.layer4_weights) + self.layer4_biases

		return output

	def choose_action(self, current_screen, total_frame):

		start_time = time.clock()

		#p_initial - (n * (p_initial - p_final)) / (total)
		if self.random_action_porb > self.random_pick_p_end:
			self.random_action_porb = self.random_action_porb - \
		                          (self.random_pick_p_start - self.random_pick_p_end)/self.random_pick_peiriod

		logger.log_write_info('random_action_porb = ' + str(self.random_action_porb))

		if np.random.rand() < self.random_action_porb:
			nextaction = np.random.randint(0, 5)
			logger.log_write_debug(' dqn, choose action rondomly, need time ' + str(time.clock() - start_time))
			logger.log_write_info('random action ' + str(nextaction))
			return [nextaction]
		else:
			nextaction = tf.argmax(self.model(self.current_screen), 1)
			logger.log_write_info('dqn select action ' + str(nextaction))
			logger.log_write_debug(' dqn, choose action by DQN, need time ' + str(time.clock() - start_time))
			return self.s.run([nextaction], {self.current_screen : current_screen})

	def training_computation(self):
		s0_norm = tf.nn.l2_normalize(self.s0, 1, epsilon=1e-12, name=None)
		#self.q = self.model(self.s0)
		self.q = self.model(s0_norm)

		s1_norm = tf.nn.l2_normalize(self.s1, 1, epsilon=1e-12, name=None)
		#self.q_max = tf.reduce_max(self.model(self.s1), 1, keep_dims=True)
		self.q_max = tf.reduce_max(self.model(s1_norm), 1, keep_dims=True)

		discount_q = self.discount_rate * self.q_max

		self.y = self.rewards + discount_q

		#self.error = tf.square(tf.sub(self.y, self.q))
		error = tf.square(tf.sub(self.y, self.q_real))
		#self.optmizer = tf.train.RMSPropOptimizer(learning_rate= 0.001, decay=0.9).minimize(self.error)

		optimizer_op = tf.train.RMSPropOptimizer(learning_rate=0.00001, decay=0.9).minimize(error)

		#optimizer_op = self.optimizer.minimize(error)

		# Optimizer.
		return error, optimizer_op


	def dqn_training(self, db_manager, frame_num):

		s0, actions, rewards, s1 = db_manager.get_minibach_smaple()

		s1_array = np.array(s1)

		if os.listdir('./ckp') != []:
			if frame_num == 3200:
				self.saver.restore(sess=self.s, save_path='./ckp/dqn')
				print('***************************check point loaded*******************************************')
				logger.log_write_info('***************************check point loaded*******************************************')

		q = self.s.run(self.q,{self.s0 : s0})

		for i in range(len(actions)):
			l = q[i]
			a = actions[i][0]
			if a == -1 or a == None:
				a = np.random.randint(low=0,high=4)
			self.q_result[i][0] = l[a]

		error, optimizer, l3_w, l4_w, q_max, y = self.s.run(
				[
					self.error,
					self.optimizer_op,
					self.layer3_weights,
					self.layer4_weights,
					self.q_max,
					self.y
				],
				{
					self.q_real : self.q_result,
					self.s1     : s1,
					self.actions: actions,
					self.rewards: rewards
				})

		if frame_num > 4800 and frame_num % 20000 == 0:
			print 'check point saved '
			logger.log_write_info('check point saved ')
			self.saver.save(sess=self.s,
			                save_path='./ckp/dqn',
			                #global_step=frame_num,
			                latest_filename='latest_ckp')

		logger.log_write_info('q_result %f' +str(self.q_result))
		logger.log_write_info('q_max %f' +str(q_max))
		logger.log_write_info('y %f' +str(y))
		logger.log_write_info('training error  = ' + str(error))

		#print 'layer1_weights ', layer1_weights
		#print 'layer2_weights ', layer2_weights
		#print 'layer3_weights ', layer3_weights
		#print 'layer4_weights ', layer4_weights
