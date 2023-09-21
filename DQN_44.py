# Test basic movement 44: Collision Avoidance
# RL Framework: Deep Q-learning
# Depth image as a state
# Saved model: trained_DQN_PC_44.h5
# Action spaces: 4 Dimensions [Forward, Left, Right, Backward]
# Note: Use DepthImage (Gray-scale), New reward shaping function, Use gradient method instead of fitting.
# Trained episode: 1500
# self.step = 102400 (Full)


import airsim  # Import airsim API
import numpy as np  # Import numpy
import math  # Import math for Python
from random import random, randint, choice, randrange
from time import sleep  # Import sleep for delaying

# ******************************************* Keras, Tensorflow library declaration ************************************
import tensorflow as tf  # Tensorflow
from keras.layers import Conv2D, Dense, BatchNormalization, Activation, Input, MaxPool2D, Flatten, Concatenate
from keras.models import Sequential, load_model, Model
from keras.optimizers import RMSprop, Adam
# ******************************************* Keras, Tensorflow library declaration ************************************


# ******************************************* Deep Q learning class start **********************************************
class DQLearning:
    def __init__(self):
        # Initialise Airsim Client
        self.drone = airsim.MultirotorClient()
        self.drone.confirmConnection()
        self.drone.reset()
        self.drone.enableApiControl(True)
        self.drone.armDisarm(True)
        self.initial_pose = self.takeoff(2)  # Get initial pose of the drone (Will be used for starting the new episode)

        # Initialise Map
        self.map_x = 100
        self.map_y = 100
        self.min_depth_meter = 0  # Minimum distance of depth camera
        self.max_depth_meter = 50  # Maximum distance of depth camera

        # Initialise State Parameters
        self.death_flag = False  # Dead indicator
        self.terminated = False  # Terminated indicator
        self.buffer_size = 102400  # Experience Replay Buffer size
        self.batch_buffer_size = 128  # Sampling batch size
        self.actions = [0, 1, 2, 3]  # Action space --> [0=Forward, 1=Left, 2=Right, 3=Backward]
        self.distance_target_dimension = 1  # <-- Number of progression in percent
        # Current state parameters
        self.state = []  # S_t
        self.progression = np.zeros([self.distance_target_dimension], dtype=float)  # Progression variable
        # Next state parameters
        self.next_state = self.state  # S_{t+1}
        self.next_progression = np.zeros([self.distance_target_dimension], dtype=float)  # Next progression variable
        self.reward = 0  # R_{t}
        self.step = 1  # Counter for storing the experiences
        self.img_width = 84  # state image width
        self.img_height = 84  # state image height
        self.img_depth = 1  # <-- Image data type (1 = Black and White), (3 = RGB)
        self.append_reward = []  # Variable to store episode reward.
        self.prev_dis = 0  # Previous distance between the destination and the agent.
        self.indices = np.zeros([self.batch_buffer_size, 2])  # Indices variable for slice the Q-value
        self.indices[:, 0] = np.arange(self.batch_buffer_size)  # First index which indicates the number of batch size.
        self.grad_glob_norm = 0  # Global gradient <-- Just to visualise the gradient of the network.
        self.cost = 0  # Value from cost function

        # Get the initial latitude and longitude
        self.lat0 = self.drone.getMultirotorState().gps_location.latitude  # Get the initial latitude
        self.lon0 = self.drone.getMultirotorState().gps_location.longitude  # Get the initial longitude
        self.alt0 = self.drone.getMultirotorState().gps_location.altitude  # Get the initial altitude
        self.drone_coordinates = self.coordinates()  # Current coordinates of the drone
        self.next_drone_coordinates = self.drone_coordinates  # Next coordinates of the drone

        # Initialise the Target position
        self.final_coordinates = [100, 0]  # Coordinate of the destination
        self.allowed_y = 5  # Allowed y value for the agent.
        self.max_dis = np.sqrt((self.final_coordinates[0]**2)+(self.final_coordinates[1]**2))  # Compute for max distance.

        # Create Experience Relay Buffer [S_t, a_t, death_flag, R_{t+1}, S_{t+1}]
        self.current_state_img = np.zeros([self.buffer_size, self.img_height, self.img_width, self.img_depth], dtype=float)  # Buffer of state
        self.action_data = np.zeros([self.buffer_size], dtype=int)  # Buffer of action
        self.death_flag_data = np.zeros([self.buffer_size], dtype=bool)  # Buffer of death_flag
        self.reward_data = np.zeros([self.buffer_size], dtype=float)  # Buffer of reward
        self.next_state_img = np.zeros([self.buffer_size, self.img_height, self.img_width, self.img_depth])  # Buffer of next state

        # Load the Saved Experienced Replay Buffer (erb) if continue training.
        # loaded_erb = np.load('ERB_DQN_44.npz')
        # self.current_state_img = loaded_erb['arr_0']
        # self.action_data = loaded_erb['arr_1']
        # self.death_flag_data = loaded_erb['arr_2']
        # self.reward_data = loaded_erb['arr_3']
        # self.next_state_img = loaded_erb['arr_4']

        # Initialise Network Parameters
        self.q_predict = self.action_value_network()  # <-- Action-Value Network (Predict)
        # self.q_predict = load_model('Trained_Models/trained_DQN_45.h5')  # Load last saved model if continue training.
        self.q_target = self.action_value_network()  # <-- Action-Value Network (Target)
        self.q_target.set_weights(self.q_predict.get_weights())
        self.update_count = 0  # Counter For Updating The Target Network
        self.update_period = 100  # Period For The Counter
        self.full_flag = True  # This variable indicates that the ERB is already full or not.
        self.fit_step = 128  # Defined Period For Fit The Prediction Network
        self.training_record = 1500 + 1  # Training record
        self.epoch = 5  # Gradient updating epoch times

    # Taking Off Method
    def takeoff(self, delay):
        self.drone.takeoffAsync().join()
        sleep(delay)
        return self.drone.simGetVehiclePose()

    # Action-Value Network Creation Method
    def action_value_network(self):
        # Define hyperparameters
        num_filter = 64
        filter_width = 12
        filter_height = 12

        # Input_1: Image Input
        img_input = Input(shape=(self.img_height, self.img_width, self.img_depth))

        # CNN-1
        cnn_1 = Conv2D(num_filter, kernel_size=(filter_width, filter_height), strides=(1, 1), padding='same')(img_input)
        cnn_1 = BatchNormalization()(cnn_1)
        cnn_1 = Activation('relu')(cnn_1)
        cnn_1 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(cnn_1)

        # CNN-2
        cnn_2 = Conv2D(num_filter*2, kernel_size=(filter_width//2, filter_height//2), strides=(2, 2), padding='same')(cnn_1)
        cnn_2 = BatchNormalization()(cnn_2)
        cnn_2 = Activation('relu')(cnn_2)
        cnn_2 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(cnn_2)

        # CNN-3
        cnn_3 = Conv2D(num_filter*2, kernel_size=(filter_width//2, filter_height//2), strides=(2, 2), padding='same')(cnn_2)
        cnn_3 = BatchNormalization()(cnn_3)
        cnn_3 = Activation('relu')(cnn_3)
        cnn_3 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(cnn_3)

        # Flatten layer for the CNNs
        flatten_1 = Flatten()(cnn_3)
        img_dense_1 = Dense(128)(flatten_1)
        img_dense_2 = Dense(128)(img_dense_1)

        # Output layers
        outputs = Dense(len(self.actions), activation='linear')(img_dense_2)

        # Define the model
        model = Model(inputs=img_input, outputs=outputs, name='action_value_model')
        model.compile(optimizer=Adam(learning_rate=0.0001, clipnorm=1))
        return model

    # Action Space Method
    def action_space(self, action_no):
        # Define speed and duration
        throttle = 1  # m/s
        duration = 1  # s
        # Lift compensator to keep the drone in the air
        if self.drone.getMultirotorState().gps_location.altitude < 123.66:
            lift_compensation = -0.25
        elif self.drone.getMultirotorState().gps_location.altitude > 125:
            lift_compensation = 0.25
        else:
            lift_compensation = 0

        # Action Space
        if action_no == 0:
            # Forward --> move +x
            self.drone.moveByVelocityAsync(throttle, 0, lift_compensation, duration).join()
        elif action_no == 1:
            # Left --> move -y
            self.drone.moveByVelocityAsync(0, -throttle, lift_compensation, duration).join()
        elif action_no == 2:
            # Right --> move +y
            self.drone.moveByVelocityAsync(0, throttle, lift_compensation, duration).join()
        elif action_no == 3:
            # Backward --> move -x
            self.drone.moveByVelocityAsync(-throttle, 0, lift_compensation, duration).join()
        else:
            # No action
            self.drone.moveByVelocityAsync(0, 0, 0, duration).join()

    # Environment Method ------ Action Taking and Reward Shaping
    def environment(self, action):
        # Take action
        self.action_space(action)

        # Get the drone coordinates after taking and action (next coordinates)
        self.next_drone_coordinates = self.coordinates()

        # Progression Reward
        # Finding the euclidean distance between object and the drone.
        dis = np.sqrt((self.final_coordinates[0] - self.next_drone_coordinates[0]) ** 2 +
                      (self.final_coordinates[1] - self.next_drone_coordinates[1]) ** 2)
        progress = 100 - (100 * dis / self.max_dis)  # The progression in percent

        # Travel distance is always going to be positive value.
        travel_dis = np.sqrt((self.next_drone_coordinates[0] - self.drone_coordinates[0]) ** 2 +
                             (self.next_drone_coordinates[1] - self.drone_coordinates[1]) ** 2)

        # Check the previous distance and the current distance between the drone and the target
        if dis < self.prev_dis - 0.1:
            travel_dis = travel_dis * 10
        else:
            travel_dis = travel_dis * (-10)

        # Total reward
        r = travel_dis

        # Check if the drone is moving out of the map
        if self.next_drone_coordinates[0] < -0.1 or abs(self.next_drone_coordinates[1]) > self.map_y:
            r = -100
            self.death_flag = True

        # Check The Collision (If true --> Dead - Penalised)
        if self.next_drone_coordinates[0] > 1:
            if self.drone.simGetCollisionInfo().has_collided:
                r = -100
                self.death_flag = True

        # Check The Destination (If true --> Terminated - Rewarded)
        if self.next_drone_coordinates[0] > self.final_coordinates[0]:
            self.death_flag = True
            if self.final_coordinates[1] - self.allowed_y <= self.next_drone_coordinates[1] \
                    <= self.final_coordinates[1] + self.allowed_y:
                r = 100
            else:
                r = r

        # Reward clipping
        if r > 100:
            r = 100
        elif r < -100:
            r = -100

        # Get The Image (Next state S_{t+1})
        get_img_str, = self.drone.simGetImages([airsim.ImageRequest("0", airsim.ImageType.DepthPerspective, True, False)])
        depth_img = airsim.list_to_2d_float_array(get_img_str.image_data_float, get_img_str.width, get_img_str.height)
        depth_img = depth_img.reshape(get_img_str.height, get_img_str.width, 1)
        depth_img = np.interp(depth_img, (self.min_depth_meter, self.max_depth_meter), (0, 255))
        self.next_state = depth_img

        # Get the next difference distance between the target and the drone.
        self.next_progression = progress
        self.prev_dis = dis
        return r

    # Coordinate Conversion Method
    def coordinates(self):
        lat = self.drone.getMultirotorState().gps_location.latitude
        lon = self.drone.getMultirotorState().gps_location.longitude
        lat0rad = math.radians(self.lat0)
        mdeg_lon = (111415.13 * np.cos(lat0rad) - 94.55 * np.cos(3 * lat0rad) - 0.12 * np.cos(5 * lat0rad))
        mdeg_lat = (111132.09 - 566.05 * np.cos(2 * lat0rad) + 1.2 * np.cos(4 * lat0rad) - 0.002 * np.cos(6 * lat0rad))

        x = (lat - self.lat0) * mdeg_lat
        y = (lon - self.lon0) * mdeg_lon

        return [x, y]

    # ------------------------------------------- Networks Training Start ----------------------------------------------
    # Network Training Method
    def action_value_network_training(self, indices):
        # Train the network by fit the new input and output features
        gamma = 0.99  # Discount factor

        # Store the transitions in the batch experienced replay buffer.
        current_state_img_batch = np.zeros([self.batch_buffer_size, self.img_height, self.img_width, self.img_depth], dtype=float)
        next_state_img_batch = np.zeros([self.batch_buffer_size, self.img_height, self.img_width, self.img_depth], dtype=float)
        action_append = np.zeros([self.batch_buffer_size], dtype=int)
        y_t = np.zeros([self.batch_buffer_size, len(self.actions)], dtype=float)
        for j in range(self.batch_buffer_size):
            current_state_img_batch[j] = self.current_state_img[indices[j]]
            next_state_img_batch[j] = self.next_state_img[indices[j]]
            action_append[j] = self.action_data[indices[j]]

            # Get the output features data (y_t) (Expected Future Return: Deep Q Network Algorithm)
            # Check if it's terminated ?
            if self.death_flag_data[indices[j]]:
                # If terminated --> y_t = reward_{t+1}
                y_t[j, self.action_data[indices[j]]] = self.reward_data[indices[j]]
            else:
                # If not terminated --> y_t = reward_{t+1} + (gamma * max Q_target(S_{t+1}, a_t, \theta_bar))
                next_state_img_predicted = next_state_img_batch[j]
                next_state_img_predicted = np.expand_dims(next_state_img_predicted, axis=0)
                next_state_img_predicted_tensor = tf.convert_to_tensor(next_state_img_predicted)

                y_t[j, self.action_data[indices[j]]] = self.reward_data[indices[j]] + (gamma * np.max(
                                                       self.q_target(next_state_img_predicted_tensor).numpy()[0]))

        # ---------------------------------- Find the gradient of the cost function ------------------------------------
        # ---------------------------------------- Prepare the state data ----------------------------------------------
        # Convert the numpy array to tensor format in order to compute the gradient in the tensorflow's gradient tape
        current_state_img_batch_tensor = tf.convert_to_tensor(current_state_img_batch, dtype=tf.float32)
        y_true_tensor = tf.convert_to_tensor(y_t, dtype=tf.float32)
        self.indices[:, 1] = action_append  # Second index of slice variable is the actions that sampled from ERB.

        # ------------------------------------- Tensorflow's Gradient Tape ---------------------------------------------
        # Compute the gradient in loop for self.epoch times
        for m in range(self.epoch):
            with tf.GradientTape() as tape:
                tape.watch(current_state_img_batch_tensor)
                tape.watch(y_true_tensor)
                y_predict_tensor = self.q_predict(current_state_img_batch_tensor)
                # Cost function <-- Mean Square Error
                # Use the tf.gather_nd to slice only the Q-value of selected action from ERB.
                # Update only the Q-value from the selected action
                cost = tf.keras.losses.MSE(tf.gather_nd(y_true_tensor, indices=self.indices.astype(int)),
                                           tf.gather_nd(y_predict_tensor, indices=self.indices.astype(int)))
            # Compute for the gradients
            self.cost = cost.numpy()
            cost_gradient = tape.gradient(cost, self.q_predict.trainable_variables)
            glob_gradient = tf.linalg.global_norm(cost_gradient)  # Compute the global gradient norm (just to visualise)
            self.grad_glob_norm = glob_gradient.numpy()
            # Apply Gradients
            self.q_predict.optimizer.apply_gradients(zip(cost_gradient, self.q_predict.trainable_variables))
        self.q_predict.save("Trained_Models/trained_DQN_44.h5")  # Save the model

        # Update Target Network's Weights
        self.update_count += 1
        if self.update_count >= self.update_period:
            self.q_target.set_weights(self.q_predict.get_weights())
            self.update_count = 0
    # -------------------------------------------- Networks Training End -----------------------------------------------

    # ---------------------------------------------- Main loop start ---------------------------------------------------
    def agent_training(self, episode):
        epsilon = 0.1  # Initial epsilon value = 1, minimum = 0.1
        for i in range(episode):
            self.drone.reset()  # Reset the drone
            self.drone.enableApiControl(True)  # Enable API control for airsim
            # Initialise the starting location of the drone
            pose = self.drone.simGetVehiclePose()
            init_y = randrange(10, 90)  # Random y position in the environment
            pose.position.y_val = init_y
            self.drone.simSetVehiclePose(pose, False)
            self.takeoff(0)  # Take off the drone

            # Initialise the target location
            self.final_coordinates = [100, 0]
            self.final_coordinates[1] = init_y

            # Reset death_flag and state
            self.death_flag = False  # Death flag
            self.terminated = False  # Terminated flag
            fit = False  # Fit flag
            sum_episode_reward = 0  # Cumulative reward for each episode
            step_count = 0  # step count in one episode

            # ----------------------------------- Prepare current state input (S_t) start ------------------------------
            # Current state 1 (S_{t_1})
            # Get The Depth Image (84 x 84 x 1) (Gray scale)
            get_img_str, = self.drone.simGetImages([airsim.ImageRequest("0", airsim.ImageType.DepthPerspective, True, False)])
            depth_img = airsim.list_to_2d_float_array(get_img_str.image_data_float, get_img_str.width, get_img_str.height)
            depth_img = depth_img.reshape(get_img_str.height, get_img_str.width, 1)
            depth_img = np.interp(depth_img, (self.min_depth_meter, self.max_depth_meter), (0, 255))
            self.state = depth_img

            # Get the initial location of the agent
            self.drone_coordinates = self.coordinates()
            # Compute for the euclidean distance of the drone
            distance_target = np.sqrt((self.final_coordinates[0] - self.drone_coordinates[0])**2 +
                                      (self.final_coordinates[1] - self.drone_coordinates[1])**2)
            self.max_dis = distance_target  # Set max distance between the drone and the target.
            self.prev_dis = distance_target  # Initialise the distance between the drone and the target.
            self.progression = 100 - (100 * distance_target / self.max_dis)  # Compute the progression.
            # ------------------------------------ Prepare current state input (S_t) end -------------------------------

            # Adaptive Epsilon Value
            epsilon = epsilon - 0.002  # Epsilon decays 0.002 for every episode.
            if epsilon < 0.1:
                epsilon = 0.1  # Allowed minimum epsilon value = 0.1

            # ------------------- Loop for the operations (Time step = self.batch_buffer_size) -------------------------
            while not self.terminated:
                # Select the action by epsilon greedy policy
                rand = random()
                if rand <= epsilon:
                    # Random any actions
                    selected_action = choice(self.actions)
                else:
                    # Select the max Q-value in the Q(S,a)
                    img_state = np.expand_dims(self.state, axis=0)
                    img_state_tensor = tf.convert_to_tensor(img_state)
                    q_predicted = self.q_predict(img_state_tensor)
                    max_q = np.unravel_index(np.argmax(q_predicted.numpy()[0]), q_predicted.shape)
                    selected_action = max_q[1]

                # Environment <-- Reward shaping
                self.reward = self.environment(selected_action)
                sum_episode_reward = sum_episode_reward + self.reward

                # ------------- Store the tuples (S_t, a_t, death_flag, r_{t+1}, S_{t+1}) in the Experience Relay Buffer
                # Check if the step count is over the buffer size, reset to zero and set the full flag.
                if self.step > (self.buffer_size - 1):
                    self.step = 0
                    self.full_flag = True
                # Store in the buffers
                self.current_state_img[self.step] = self.state
                self.action_data[self.step] = selected_action
                self.death_flag_data[self.step] = self.death_flag
                self.reward_data[self.step] = self.reward
                self.next_state_img[self.step] = self.next_state

                # Check fit flag --> If the step divided by the fit_step no. equals to zero --> set fit_flag 'True'
                if self.step % self.fit_step == 0:
                    if not self.full_flag and self.step == 0:
                        # If the ERB is not full yet and the step count is 0, do not fit. (fit = false)
                        fit = False
                    else:
                        # If the ERB is already full and the step count is reached, terminated and fit (fit = True)
                        fit = True

                # # Save the buffer every 5,000 steps
                if self.step % 5000 == 0:
                    np.savez('ERB_DQN_44', self.current_state_img,
                             self.action_data, self.death_flag_data, self.reward_data,
                             self.next_state_img)

                # Step increment
                self.step += 1
                step_count += 1

                # Update the state
                if self.death_flag or step_count >= self.fit_step:
                    # If the next state is death, or already over 128 steps --> terminated.
                    self.terminated = True
                    if self.step >= self.batch_buffer_size or self.full_flag:
                        # Start to learn the model only when the data in ERB are more than batch buffer size.
                        fit = True
                else:
                    # If the next state is not death, continue to the next step.
                    self.state = self.next_state  # Get the next image as current img
                    self.progression = self.next_progression  # Get the next distance target as current
                    self.drone_coordinates = self.next_drone_coordinates  # Get the next location as current location

            # Store Episode's Reward
            self.append_reward.append(sum_episode_reward)
            np.save('Append_Reward/append_reward_44', self.append_reward)

            # Save the model every 1000 episodes
            if (self.training_record + i) % 100 == 0:
                self.q_predict.save('Trained_Models/DQN_44_{:d}.h5'.format(self.training_record + i))

            # If the fit flag is true --> Train The Network
            if fit:
                if not self.full_flag:
                    # If the ERB is not full, only sampling from the size within the step count number.
                    # Shuffle the indices of buffer
                    random_indices = np.arange(self.step)
                    np.random.shuffle(random_indices)
                    # Update the action-value network after the episode is terminated.
                    self.action_value_network_training(random_indices)
                else:
                    # If the ERB is full already, sampling from the whole size ERB.
                    # Shuffle the indices of buffer
                    random_indices = np.arange(self.buffer_size)
                    np.random.shuffle(random_indices)
                    # Update the action-value network after the episode is terminated.
                    self.action_value_network_training(random_indices)

            # Print the status of Learning.
            print('Episode: ', i + self.training_record, ', Step: ', self.step, ', Sum_reward: ', sum_episode_reward,
                  ', Avg_reward: ', np.sum(self.append_reward)/len(self.append_reward), ', Progression: ',
                  self.progression, ', Loss: ', self.cost, ', Gradient: ', self.grad_glob_norm)
    # ---------------------------------------------- Main loop end -----------------------------------------------------
# ******************************************* Deep Q learning class end ************************************************


# ******************************************* Main Program Start *******************************************************
# Training the drone
if __name__ == "__main__":
    droneDQL = DQLearning()  # <-- Create drone Deep Q-learning object
    droneDQL.agent_training(1500)  # <-- Training the drone (episode)

    droneDQL.q_predict.summary()
    droneDQL.q_predict.save("Trained_Models/trained_DQN_44.h5")
# ******************************************* Main Program End *********************************************************
