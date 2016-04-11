import cv2 as cv
import os
import numpy as np
import time
import logger

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

def init_database():

	if not os.path.exists('./smaple'):
		os.makedirs('./smaple')

def save(frame_num, img):

	global current_frame_num

	'''
	1. generate path for an input sample
	2. resize the image
	3. store the down-sampled image to the path
	'''

	'''

	file_path = generate_path(frame_num)

	dst = cv.resize(img, dsize=(84,84))
	cv.imwrite(file_path, dst)
	'''

	start_time = time.clock()
	dst = cv.resize(img, dsize=(84,84))


	if len(memory) == replaymem_size:
		memory.pop()

	memory.append(dst)

	logger.log_write_info('store one sample needs time ' + str(time.clock() - start_time))
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
	logger.log_write_info('get one minibatch states needs time ' + str(time.clock() - start_time))
	#minibach_data = memory_buffer[memory_buffer_index : memory_buffer_index + minibatch_length]

	#memory_buffer_index = memory_buffer_index + minibatch_length

	return minibatch_s0, minibatch_actions, minibatch_rewards, minibatch_s1

def add_action(a):
	logger.log_write_info('frame =' + str(current_frame_num) + ', record action ' + str(a) + ', actions recorded = ' + str(len(actions)))
	actions.append(a)

def add_reward(r):
	logger.log_write_info('frame =' + str(current_frame_num) + ', record reward ' + str(r) + ', rewards recorded = ' + str(len(rewards)))
	rewards.append(r)
