import cityflow
from time import time
import math
import numpy as np

class CityFlowEnvM(object):
    def __init__(self,
                 lane_phase_info,
                 intersection_id,
                 num_step=3600,
                 thread_num=1,
                 eta = 0.1,
                 cityflow_config_file='example/config_1x2.json',
                 # traffic_env_conf = {},
                 dataset='hangzhou',
                 path_to_log='result',
                 ):
        self.eng = cityflow.Engine(cityflow_config_file, thread_num=thread_num)
        self.num_step = num_step
        self.intersection_id = intersection_id  # list, [intersection_id, ...]
        self.state_size = None
        self.eta = eta
        self.dataset = dataset

        self.path_to_log = path_to_log

        self.info_functions = {
            "vehicles": (lambda: self.eng.get_vehicles(include_waiting=True)),
            "lane_count": self.eng.get_lane_vehicle_count,
            "lane_waiting_count": self.eng.get_lane_waiting_vehicle_count,
            "lane_vehicles": self.eng.get_lane_vehicles,
            "time": self.eng.get_current_time,
            "vehicle_distance": self.eng.get_vehicle_distance,
            "vehicle_speed":self.eng.get_vehicle_speed,
            "avg_travel_time":self.eng.get_average_travel_time,
            "pressure": self.get_pressure,
            "waiting_time_count": self.get_vehicle_waiting_time
        }
        self.fns = []
        self.info = {}
        self.vehicle_waiting_time = {}

        self.lane_intensity = {}
        ###
        self.phase_list = {}
        self.current_phase = {}
        self.current_phase_time = {}
        self.start_lane = {}
        self.end_lane = {}
        initial_phase = {}
        self.inter_neighbor = {}
        self.list_lanes = {}
        for id_ in self.intersection_id:
            self.start_lane[id_] = lane_phase_info[id_]['start_lane']
            self.end_lane[id_] = lane_phase_info[id_]['end_lane']
            self.phase_list[id_] = lane_phase_info[id_]["phase"]
            self.current_phase[id_] = self.phase_list[id_][0]
            self.current_phase_time[id_] = 0
            self.lane_intensity[id_] = {}
            self.lane_intensity[id_]['start'] = [0 for _ in self.start_lane[id_]]
            self.lane_intensity[id_]['end'] = [0 for _ in self.end_lane[id_]]
            initial_phase[id_] = 1
            ###
            inters = str.split(id_, '_')
            row = int(inters[1])
            column = int(inters[2])
            self.inter_neighbor[id_]  = []
            self.inter_neighbor[id_].append([row - 1, column])
            self.inter_neighbor[id_].append([row, column - 1])
            self.inter_neighbor[id_].append([row + 1, column])
            self.inter_neighbor[id_].append([row, column + 1])
            self.list_lanes[id_] = self.start_lane[id_] + self.end_lane[id_]
        self.get_state(initial_phase)

    def intersection_info(self, id_):
        state = {}
        state['lane_vehicle_count'] = self.eng.get_lane_vehicle_count()
        state['lane_waiting_vehicle_count'] = self.eng.get_lane_waiting_vehicle_count()
        state['lane_vehicles'] = self.eng.get_lane_vehicles()
        state['vehicle_speed'] = self.eng.get_vehicle_speed()
        state['start_lane_vehicle_count'] = {lane: state['lane_vehicle_count'][lane] for lane in self.start_lane[id_]}
        state['end_lane_vehicle_count'] = {lane: state['lane_vehicle_count'][lane] for lane in self.end_lane[id_]}
        state['start_lane_waiting_vehicle_count'] = {lane: state['lane_waiting_vehicle_count'][lane] for lane in
                                                     self.start_lane[id_]}
        state['end_lane_waiting_vehicle_count'] = {lane: state['lane_waiting_vehicle_count'][lane] for lane in
                                                   self.end_lane[id_]}
        state['start_lane_vehicles'] = {lane: state['lane_vehicles'][lane] for lane in self.start_lane[id_]}
        state['end_lane_vehicles'] = {lane: state['lane_vehicles'][lane] for lane in self.end_lane[id_]}
        state['current_phase'] = self.current_phase[id_]
        state['current_phase_time'] = self.current_phase_time[id_]
        return state

    def reset(self):
        self.eng.reset()

    def get_current_time(self):
        return self.eng.get_current_time()

    def get_lanepressure(self, id_, lane, distances):
        L = 300
        if self.dataset == 'jinan':
            if lane[-3] == '0' or lane[-3] == '2':
                L = 400
            else:
                L = 800
        elif self.dataset == 'hangzhou':
            if lane[-3] == '0' or lane[-3] == '2':
                L = 800
            else:
                L = 600
        elif self.dataset == '3x3' or self.dataset == '4x4':
            if lane[-3] == '0' or lane[-3] == '2':
                L = 300
            else:
                L = 300
        elif self.dataset == 'ny48':
            if lane[-3] == '0' or lane[-3] == '2':
                L = 350
            else:
                L = 100
        lane_pressure = 0
        sigma = 1.5
        max_speed = 11.111
        vehicles = self.eng.get_lane_vehicles()[lane]
        if lane in self.start_lane[id_]:
            for v in vehicles:
                x = distances[v]
                lane_pressure += float(format(math.log(
                x / L * sigma * (max_speed - float(self.eng.get_vehicle_info(v)["speed"])) / (
                        float(self.eng.get_vehicle_info(v)["speed"]) + 1) + 1), '.4f'))
        else:
            for v in vehicles:
                x = distances[v]
                lane_pressure += float(format(math.log(
                (L - x) / L * sigma * (max_speed - float(self.eng.get_vehicle_info(v)["speed"])) / (
                        float(self.eng.get_vehicle_info(v)["speed"]) + 1) + 1), '.4f'))
        return lane_pressure

    def step(self, action_phase, cur_step=None):
        for id_, a in action_phase.items():
            if self.current_phase[id_] == a:
                self.current_phase_time[id_] += 1
            else:
                self.current_phase[id_] = a
                self.current_phase_time[id_] = 1
            self.eng.set_tl_phase(id_, self.current_phase[id_])
        self.eng.next_step()
        self._update_infos()
        if cur_step % 5 == 4:
            distances = self.eng.get_vehicle_distance()
            for id_ in self.intersection_id:
                self.lane_intensity[id_]['start'] = [self.get_lanepressure(id_, lane, distances) for lane in self.start_lane[id_]]
                self.lane_intensity[id_]['end'] = [self.get_lanepressure(id_, lane, distances) for lane in self.end_lane[id_]]
        reward = self.get_reward()

        return self.get_state(action_phase), reward

    def get_state(self, action_phase=None):
        state = {id_: self.get_state2_(id_, action_phase[id_]) for id_ in self.intersection_id}
        return state

    def get_reward(self):
        reward = {id_: self.get_reward2_(id_) for id_ in self.intersection_id}
        return reward

    def get_timing_(self, id_, phase):
        state = self.intersection_info(id_)
        start_vehicle_count = [state['start_lane_vehicle_count'][lane] for lane in self.start_lane[id_]]
        start_vehicle_count_cop = []
        for i in range(len(start_vehicle_count)):
            if i % 3 != 2:
                start_vehicle_count_cop.append(start_vehicle_count[i])

        phase_lane = [[1,7], [3,5], [0,6], [2,4]]
        w1 = 1
        w2 = 2
        max_count = max(start_vehicle_count_cop[phase_lane[phase-1][0]], start_vehicle_count_cop[phase_lane[phase-1][1]])
        min_count = min(start_vehicle_count_cop[phase_lane[phase-1][0]], start_vehicle_count_cop[phase_lane[phase-1][1]])
        vehicle_count = (min_count * w1 + max_count * w2)/(w1+w2)
        timing = math.ceil(vehicle_count/2)*5
        if timing > 25:
            timing = 25
        if timing < 5:
            timing = 5
        return vehicle_count, timing

    def get_intensity_phase(self):
        phase = {id_: self.intensity_phase_control_(id_) for id_ in self.intersection_id}
        return phase

    def intensity_phase_control_(self, id_):
        temp = []
        end_vehicle_count_avg = []
        for i in range(4):
            end_vehicle_count_avg.append(
                math.ceil(sum([self.lane_intensity[id_]['end'][j] for j in range(i * 3, i * 3 + 3)]) / 3))
        start_vehicle_count_cop = []
        index = [1, 0, 2, 1, 0, 3, 3, 2]
        for i in range(len(self.lane_intensity[id_]['start'])):
            if i % 3 != 2:
                start_vehicle_count_cop.append(self.lane_intensity[id_]['start'][i])
        for i in range(len(start_vehicle_count_cop)):
            temp.append(start_vehicle_count_cop[i] - end_vehicle_count_avg[index[i]])

        feat10 = temp[1] + temp[7]
        feat20 = temp[3] + temp[5]
        feat30 = temp[0] + temp[6]
        feat40 = temp[2] + temp[4]
        idx = np.argmax([feat10, feat20, feat30, feat40])
        return idx

    def get_length_phase(self):
        phase = {id_: self.lane_queue_length_phase_control_(id_) for id_ in self.intersection_id}
        return phase

    def lane_queue_length_phase_control_(self, id_):
        state = self.intersection_info(id_)
        feat0 = [state['start_lane_waiting_vehicle_count'][lane] for lane in self.start_lane[id_]]

        feat10 = feat0[1] + feat0[10]
        feat20 = feat0[4] + feat0[7]
        feat30 = feat0[0] + feat0[9]
        feat40 = feat0[3] + feat0[6]

        idx = np.argmax([feat10, feat20, feat30, feat40])
        return idx



    def get_state_(self, id_, action_phase = None):
        inters = str.split(id_, '_')
        row = int(inters[1])
        column = int(inters[2])
        neighbor = self.inter_neighbor[id_]
        eta = self.eta # 0.1
        intensity = []
        temp = []

        end_vehicle_count_avg = []
        for i in range(4):
            end_vehicle_count_avg.append(math.ceil(sum([self.lane_intensity[id_]['end'][j] for j in range(i * 3, i * 3 + 3)]) / 3))
        start_vehicle_count_cop = []
        index = [1, 0, 2, 1, 0, 3, 3, 2]
        for i in range(len(self.lane_intensity[id_]['start'])):
            if i % 3 != 2:
                start_vehicle_count_cop.append(self.lane_intensity[id_]['start'][i])
        for i in range(len(start_vehicle_count_cop)):
            temp.append(start_vehicle_count_cop[i] - end_vehicle_count_avg[index[i]])

        intensity.append(temp[1] + temp[7])
        intensity.append(temp[3] + temp[5])
        intensity.append(temp[0] + temp[6])
        intensity.append(temp[2] + temp[4])
        return_state = intensity

        for e in neighbor:
            interid = inters[0] + '_' + str(e[0]) + '_' + str(e[1])
            if interid in self.intersection_id:
                return_state.append(eta * self.get_neigh_pressure(nei_id_=interid, row=row, col=column, nei_col=e[0], nei_row=e[1], action_phase=action_phase))
            else:
                return_state.append(0)
        return np.reshape(return_state, [1, 8])

    def get_state2_(self, id_, action_phase = None):
        return_state = []
        temp = []
        end_vehicle_count_avg = []
        for i in range(4):
            end_vehicle_count_avg.append(math.ceil(sum([self.lane_intensity[id_]['end'][j] for j in range(i * 3, i * 3 + 3)]) / 3))
        start_vehicle_count_cop = []
        index = [1, 0, 2, 1, 0, 3, 3, 2]
        for i in range(len(self.lane_intensity[id_]['start'])):
            if i % 3 != 2:
                start_vehicle_count_cop.append(self.lane_intensity[id_]['start'][i])
        for i in range(len(start_vehicle_count_cop)):
            temp.append(start_vehicle_count_cop[i] - end_vehicle_count_avg[index[i]])

        return_state.append(temp[1] + temp[7])
        return_state.append(temp[3] + temp[5])
        return_state.append(temp[0] + temp[6])
        return_state.append(temp[2] + temp[4])

        state = self.intersection_info(id_)
        feat0 = [state['start_lane_waiting_vehicle_count'][lane] for lane in self.start_lane[id_]]
        return_state.append(feat0[1] + feat0[10])
        return_state.append(feat0[4] + feat0[7])
        return_state.append(feat0[0] + feat0[9])
        return_state.append(feat0[3] + feat0[6])

        return np.reshape(return_state, [1, 8])

    def get_neigh_pressure(self, nei_id_, row, col, nei_row, nei_col, action_phase = None):
        pressure = 0
        start_vehicle_count = self.lane_intensity[nei_id_]['start']

        if nei_row < row:
            if action_phase == 1:
                pressure += start_vehicle_count[5] + start_vehicle_count[1]
            if action_phase == 4:
                pressure += start_vehicle_count[5] + start_vehicle_count[6]
        elif nei_row > row:
            if action_phase == 1:
                pressure += start_vehicle_count[8] + start_vehicle_count[10]
            if action_phase == 4:
                pressure += start_vehicle_count[8] + start_vehicle_count[3]

        if nei_col < col:
            if action_phase == 2:
                pressure += start_vehicle_count[11] + start_vehicle_count[4]
            if action_phase == 3:
                pressure += start_vehicle_count[11] + start_vehicle_count[0]
        elif nei_col > col:
            if action_phase == 2:
                pressure += start_vehicle_count[2] + start_vehicle_count[7]
            if action_phase == 3:
                pressure += start_vehicle_count[2] + start_vehicle_count[9]

        return pressure

    def preprocess_state(self, state, action = None):
        return_state = np.array(state)
        if self.state_size is None:
            self.state_size = len(return_state.flatten())
        if action != None:
            return_state = np.reshape(return_state, [1, self.state_size+1])
        else:
            return_state = np.reshape(return_state, [1, self.state_size])
        return return_state



    def get_reward_(self, id_, action_phase = None):
        inters = str.split(id_, '_')
        row = int(inters[1])
        column = int(inters[2])
        neighbor = self.inter_neighbor[id_]
        eta = self.eta

        start_vehicle_count = self.lane_intensity[id_]['start']
        end_vehicle_count = self.lane_intensity[id_]['end']
        start_vehicle_count_cop = []
        end_vehicle_count_cop = []
        for i in range(len(start_vehicle_count)):
            if i % 3 != 2:
                start_vehicle_count_cop.append(start_vehicle_count[i])
                end_vehicle_count_cop.append(end_vehicle_count[i])
        intensity = sum(start_vehicle_count_cop) - sum(end_vehicle_count_cop)

        neigh_pre = 0
        for e in neighbor:
            interid = inters[0] + '_' + str(e[0]) + '_' + str(e[1])
            if interid in self.intersection_id:
                neigh_pre += eta * self.get_neigh_pressure(nei_id_=interid, row=row, col=column, nei_col=e[0], nei_row=e[1], action_phase=self.current_phase[id_])
            else:
                neigh_pre += 0
        reward = - intensity  - eta * neigh_pre
        return reward

    def get_reward2_(self, id_, action_phase = None):
        eta = self.eta
        state = self.intersection_info(id_)

        start_vehicle_count = self.lane_intensity[id_]['start']
        end_vehicle_count = self.lane_intensity[id_]['end']
        start_vehicle_count_cop = []
        end_vehicle_count_cop = []
        for i in range(len(start_vehicle_count)):
            if i % 3 != 2:
                start_vehicle_count_cop.append(start_vehicle_count[i])
                end_vehicle_count_cop.append(end_vehicle_count[i])
        intensity = sum(start_vehicle_count_cop) - sum(end_vehicle_count_cop)

        length_ = [state['start_lane_waiting_vehicle_count'][lane] for lane in self.start_lane[id_]]
        length = - np.absolute(np.sum(length_))
        reward = - intensity  - eta * length
        return reward

    def get_pressure(self):
        pressure = {id_: self.get_pressure_(id_) for id_ in self.intersection_id}
        return pressure

    def get_pressure_(self, id_):
        state = self.intersection_info(id_)
        start_vehicle_count = [state['start_lane_vehicle_count'][lane] for lane in self.start_lane[id_]]
        end_vehicle_count = [state['end_lane_vehicle_count'][lane] for lane in self.end_lane[id_]]
        pressure = sum(start_vehicle_count) - sum(end_vehicle_count)
        return pressure

    def get_quene_length_reward(self):
        reward = {id_: self.get_quene_length_reward_(id_) for id_ in self.intersection_id}
        return reward

    def get_quene_length_reward_(self, id_):
        state = self.intersection_info(id_)
        feat0 = [state['start_lane_waiting_vehicle_count'][lane] for lane in self.start_lane[id_]]
        reward = - np.absolute(np.sum(feat0))
        return reward

    def get_vehicle_waiting_time(self):
        vehicles = self.eng.get_vehicles(include_waiting=False)
        vehicle_speed = self.eng.get_vehicle_speed()
        for vehicle in vehicles:
            if vehicle not in self.vehicle_waiting_time.keys():
                self.vehicle_waiting_time[vehicle] = 0
            if vehicle_speed[vehicle] < 0.1:
                self.vehicle_waiting_time[vehicle] += 1
            else:
                self.vehicle_waiting_time[vehicle] = 0
        return self.vehicle_waiting_time

    def _update_infos(self):
        self.info = {}
        for fn in self.fns:
            self.info[fn] = self.info_functions[fn]()

    def get_info(self, info):
        return self.info[info]

    def subscribe(self, fns):
        if isinstance(fns, str):
            fns = [fns]
        for fn in fns:
            if fn in self.info_functions:
                if not fn in self.fns:
                    self.fns.append(fn)
            else:
                raise Exception("info function %s not exists" % fn)