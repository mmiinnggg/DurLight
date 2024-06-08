import random
import numpy as np
from collections import deque
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
import tensorflow as tf
from tensorflow.keras import backend as K
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def self_attention_layer(inputs):
    feat_dims = 2

    query = Dense(feat_dims, activation='relu')(inputs)
    key = Dense(feat_dims, activation='relu')(inputs)
    value = Dense(feat_dims, activation='relu')(inputs)

    energy = tf.matmul(query, key, transpose_b=True)

    scale = tf.math.sqrt(tf.cast(feat_dims, tf.float32))
    energy = energy / scale

    attention_weights = tf.nn.softmax(energy, axis=-1)

    weighted_sum = tf.matmul(attention_weights, value)

    output = tf.reshape(weighted_sum, [-1, feat_dims])

    return output

class DQNAgent(object):
    def __init__(self,
                 intersection_id,
                 state_size=8,
                 action_size=8,
                 batch_size=32,
                 phase_list=[],
                 timing_list=[],
                 mode='action',
                 time_size = 5,
                 env=None
                 ):
        self.env = env
        self.intersection_id = intersection_id
        self.action_size = action_size
        self.time_size = time_size
        self.batch_size = batch_size  # 32
        self.state_size = state_size

        self.memory_size = 2000
        self.memory = deque(maxlen=self.memory_size)

        self.learning_start = 2000
        self.update_model_freq = 1
        self.update_target_model_freq = 20

        self.gamma = 0.95  # discount rate
        self.epsilon = 0.1  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001  # 0.005
        self.d_dense = 20
        self.n_layer = 2

        self.mode = mode
        self.model = self._build_model(mode)

        self.step = 0
        self.phase_list = phase_list
        self.timing_list = timing_list

    def _build_model(self, mode='action'):

        feat1 = Input(shape=(8,))

        top_four = feat1[:4]
        max_index = tf.math.argmax(top_four, axis=-1, output_type=tf.int32)

        feat_map = [[0, 4], [1, 5], [2, 6], [3, 7]]
        lane_feats_s = tf.split(value=feat1, num_or_size_splits=8, axis=1)
        phase_feats_map = []
        for i in range(4):
            tmp_feat_1 = tf.concat([lane_feats_s[idx] for idx in feat_map[i]], axis=1)
            tmp_feat_2 = Lambda(self_attention_layer)(tmp_feat_1)
            scale_factor = tf.constant(0.8 if i == max_index else 0.2, dtype=tf.float32)
            tmp_feat_3 = tmp_feat_2 * scale_factor
            phase_feats_map.append(tmp_feat_3)

        phase_feat_all = tf.concat(phase_feats_map, axis=0)
        phase_feat_all = tf.reshape(phase_feat_all, [-1, 4 * 2])
        dense0 = Dense(20, activation="sigmoid")(phase_feat_all)

        if mode=='action':
            q_values = Dense(self.action_size, activation="linear", name="q_values")(dense0)
        elif mode=='time':
            q_values = Dense(self.time_size, activation="linear", name="q_values")(dense0)
        network = Model(inputs=feat1,
                        outputs=q_values)
        network.compile(optimizer=RMSprop(lr=self.learning_rate),
                        loss="mean_squared_error")
        # network.summary()
        return network

    def _reshape_ob(self, ob):
        return np.reshape(ob, (1, -1))

    def get_action(self, phase, ob):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        ob = self._reshape_ob(ob)
        act_values = self.model.predict([phase, ob])
        return np.argmax(act_values[0])

    def get_time(self, action, ob):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.time_size)
        time_values = self.model.predict(ob)
        return np.argmax(time_values[0])

    def sample(self):
        return random.randrange(self.action_size)

    def remember(self, ob, phase, action, reward, next_ob, next_phase):
        self.memory.append((ob, phase, action, reward, next_ob, next_phase))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        self.step += 1
        state_t = self.memory[-1][-1]
        replay_batch = random.sample(self.memory, self.batch_size)
        s_batch = np.reshape(np.array([replay[0] for replay in replay_batch]), [self.batch_size, self.state_size])
        next_s_batch = np.reshape(np.array([replay[4] for replay in replay_batch]), [self.batch_size, self.state_size])

        Q = self.model.predict(s_batch)
        Q_next = self.model.predict(next_s_batch)

        lr = 1
        for i, replay in enumerate(replay_batch):
            _, _, a, reward, state_n, _ = replay
            if (state_t == state_n).all():
                Q[i][a] = (1 - lr) * Q[i][a] + lr * reward
            else:
                Q[i][a] = (1 - lr) * Q[i][a] + lr * (reward + self.gamma * np.amax(Q_next[i]))
        self.model.fit(s_batch, Q, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load_model(self, dir="model/dqn", e=0):
        name = "dqn_agent_{}_{}.h5".format(self.intersection_id, e)
        model_name = os.path.join(dir, name)
        self.model.load_weights(model_name)

    def save_model(self, dir="model/dqn", e=0):
        name = "dqn_agent_{}_{}.h5".format(self.intersection_id, e)
        model_name = os.path.join(dir, name)
        self.model.save_weights(model_name)