import cv2 as cv
import os
import numpy as np
import time
import logger
import array

db_first_level = 1000
db_second_level = 1000

replaymem_size = 1000000
current_state_num = 0

history_lenth = 4

state_start_frame = []

stacked_data = np.array(np.zeros((84, 84, 4)))

memory_buffer = []
memory_buffer_size = 128
memory_buffer_index = 0

actions = []
rewards = []
minibatch_length = 16
minibatch_actions = np.array(np.zeros((minibatch_length,1)))
minibatch_rewards = np.array(np.zeros((minibatch_length,1)))
minibatch_s0 = []
minibatch_s1 = []

current_frame_num = 0

memory = []

state_dimension = 4

currnet_ovservation = []
following_ovservation = []
current_state = []


class State():
	def __init__(self, observation_0 =None,
	             action = None,
	             reward = None,
	             observation_1 = None,
	             frame = 0):
		self.observation_0 = observation_0
		self.action = action
		self.reward = reward
		self.observation_1 = observation_1
		self.frame = frame

def init_database():

	if not os.path.exists('./smaple'):
		os.makedirs('./smaple')

class Data_manager():

	def __init__(self):
		self.temp_state = None
		#self.current_observation = []
		self.current_observation = np.array(np.zeros((84,84,4)))
		self.following_observation = np.array(np.zeros((84,84,4)))
		self.state_length = 4
		self.current_action = -1
		self.current_reward = 0

		self.minibatch_length = 16
		self.current_init = 0
		self.memory_size = 1000
		self.memory = []

	def save(self,frame_num, img, action, reward):
		resized_img = cv.resize(img, dsize=(84,84))

		if self.temp_state == None:

			#if frame_num % self.state_length == self.state_length - 1 and self.current_init == 4:
			if self.current_init == 4:

				self.following_observation[:,:,frame_num % self.state_length] = resized_img
				if self.current_reward == 0:
					self.current_reward = reward
				self.current_action = action

				self.temp_state = State(observation_0=self.current_observation,
				                        frame=frame_num)
				logger.log_write_info('frame =' + str(frame_num)
				                      + 'current_observation done, NOT record action ' + str(action)
				                      + ', reward = ' + str(reward))
				return
			else:
				#self.current_observation.append(resized_img)
				logger.log_write_info('frame =' + str(frame_num)
				                      + ' recording current_observation no.'
				                      + str(frame_num % self.state_length))
				self.current_observation[:,:,frame_num % self.state_length] = resized_img
				self.current_init += 1
				return
		else:
			if frame_num % self.state_length == self.state_length - 1:
				self.following_observation[:,:,frame_num % self.state_length] = resized_img

				if self.current_reward == 0:
					self.current_reward = reward

				self.current_action = action
				self.temp_state.observation_1 = self.following_observation
				self.temp_state.action = self.current_action
				self.temp_state.reward = self.current_reward


				if self.temp_state.reward == 0 and np.random.rand() < 0.9:
					#if no reward, only recode 20% of the states.
					return
				else:
					if len(self.memory) == self.memory_size:
						self.memory.pop()

					self.memory.append(self.temp_state)

				logger.log_write_info('frame = ' + str(frame_num)
				                      + ' State into memory, numbers recorded ' + str(len(self.memory))
				                      + ' action = ' + str(self.temp_state.action)
				                      + ', reward = ' + str(self.temp_state.reward))


				self.temp_state = None
				self.current_observation = self.following_observation
				self.current_reward = 0
				self.current_action = -1
				self.following_observation = np.array(np.zeros((84,84,4)))
			else:
				logger.log_write_info('frame =' + str(frame_num)
				                      + ' recording following_observation no.' + str(frame_num % self.state_length))
				self.following_observation[:,:,frame_num % self.state_length] = resized_img
				if self.current_reward == 0:
					self.current_reward = reward
				self.current_action = action
				return

	def get_current_observation(self):
		return [self.current_observation]

	def get_minibach_smaple(self):

		minibatch_s0 = []
		minibatch_s1 = []
		for i in range(self.minibatch_length):
			action = -1
			while(action == -1):
				state_index = np.random.randint(0, len(self.memory))
				action = self.memory[state_index].action

			minibatch_actions[i, 0] = action
			if self.memory[state_index].reward != 0:
				logger.log_write_info(' picked a none-zero reward state !')
			minibatch_rewards[i, 0] = self.memory[state_index].reward

			minibatch_s0.append(self.memory[state_index].observation_0)
			minibatch_s1.append(self.memory[state_index].observation_1)

		return minibatch_s0, minibatch_actions, minibatch_rewards, minibatch_s1


def save(frame_num, img, action, reward):

	global current_frame_num
	start_time = time.clock()

	if len(currnet_ovservation) <= state_dimension:
		dst = cv.resize(img, dsize=(84,84))
		currnet_ovservation.append((dst))



	logger.log_write_info('store one sample needs time ', time.clock() - start_time)
	current_frame_num = frame_num

def get_stacked_input(state_index):

	for i in range(history_lenth):
		#name = frame_to_file(state_index * 4 + (history_lenth - 1 - i))
		#print name
		#stacked_data[:, :, i] = cv.imread(name, 0)

		memory_index = (state_index - 1) * 4 + i + 1

		#print state_index

		#print len(memory)

		#print memory_index
		stacked_data[:, :, i] = memory[memory_index]

	return stacked_data

def generate_path(frame_num):
	first_dir = frame_num / db_first_level
	dir = './samples/' + str(first_dir)
	if not os.path.exists(dir):
		os.makedirs(dir)
	img_name = frame_num % db_first_level
	return './samples/' + str(first_dir) + '/' + str(img_name) + '.png'

def frame_to_file(frame_num):
	first_dir = frame_num / db_first_level
	img_name = frame_num % db_first_level
	return './samples/' + str(first_dir) + '/' + str(img_name) + '.png'
def get_current_screen_data():
	return [get_stacked_input(int(current_frame_num / 4))]

def fill_memory_buffer():

		global memory_buffer, current_frame_num, minibatch_actions, minibatch_rewards, minibatch_s0, minibatch_s1

		memory_buffer = []
		minibatch_s0 = []
		minibatch_s1 = []

		for i in range(minibatch_length):

			action = -1
			while(action == -1):
				state_index = np.random.randint(0, current_frame_num / 4)

				if state_index >= len(actions) or state_index >= len(rewards):
					logger.log_write_info('state_index = ' + str(state_index))
					logger.log_write_info('length of actions = ' + str(len(actions)))

				action = actions[state_index]
				minibatch_actions[i, 0] = actions[state_index]
				minibatch_rewards[i, 0] = rewards[state_index]

			stacked_data_St0 = get_stacked_input(state_index)
			stacked_data_St1 = get_stacked_input(state_index+1)

			minibatch_s0.append(stacked_data_St0)
			minibatch_s1.append(stacked_data_St1)



def get_minibach_smaple():

	'''
	if memory_buffer == [] or memory_buffer_index == memory_buffer_size:

		print 'current frame num = ', current_frame_num

		fill_memory_buffer(current_frame_num)
	'''

	start_time = time.clock()

	fill_memory_buffer()
	logger.log_write_info('get one minibatch states needs time ', time.clock() - start_time)
	#minibach_data = memory_buffer[memory_buffer_index : memory_buffer_index + minibatch_length]

	#memory_buffer_index = memory_buffer_index + minibatch_length

	return minibatch_s0, minibatch_actions, minibatch_rewards, minibatch_s1

def add_action(a):
	logger.log_write_info('frame =' + str(current_frame_num) + ', record action ' + str(a) + ', actions recorded = ' + str(len(actions)))
	actions.append(a)

def add_reward(r):
	logger.log_write_info('frame =' + str(current_frame_num) + ', record reward ' + str(r) + ', rewards recorded = ' + str(len(rewards)))
	rewards.append(r)
