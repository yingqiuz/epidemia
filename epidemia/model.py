# -*- coding: utf-8 -*-
"""
basic classes and infrastructure
"""
import numpy as np
from scipy.stats import norm


class AgentBasedModel:
    """
    functions:
        normal_alpha_syn_growth_region: growth of normal alpha-syn in regions
        normal_alpha_syn_growth_edge: mobility patterns of normal alpha-syn in edges
        normal_alpha_syn_growth_edge_discrete: a discrete version of mobility patterns, using length rather than probability to control the exits and enters
        inject_mis: inject misfolded alpha-syn into seed region. default: inject one misfolded alpha-syn into substantia nigra
        misfolded_alpha_syn_spread_region: interactions of normal and misfolded alpha-syn
        misfolded_alpha_syn_spread_edge: mobility patterns of normal/misfolded alpha-syn in edges
        misfolded_alpha_syn_spread_edge_discrete: a discrete version....
        record_to_history: record the misfolded/normal alpha-syn population at each time step

    parameters:
        v: speed (default 1)
        N_regions: total number of ROIs
        dt: default 0.01
        sconn_len: structural connectivity (length)
        sconn_den: structural connectivity (strength)
        snca: SNCA expressions
        gba: GBA expressions
        roi_size: Region size
        fconn: functional connectivity
        fcscale: weights of functional connectivity
	"""

    # constructor
    def __init__(self, v=1,
                 N_regions=42,
                 dt=0.01,
                 dist=None,
                 weight=None,
                 growth_rate=None,
                 clearance_rate=None,
                 region_size=None,
                 fconn=None,
                 fcscale=None
                 ):

        # number of regions
        self.N_regions = N_regions

        # store number of normal and misfolded proteins in regions
        self.nor, self.mis = [np.zeros((N_regions,), dtype=np.int)] * 2
        self.nor_history, self.mis_history = [np.empty((0, N_regions), dtype=np.int)] * 2

        # store number of normal and misfolded proteins in paths
        self.sconn_len = np.int_(np.round(sconn_len / v))
        (self.idx_x, self.idx_y) = np.nonzero(self.sconn_len)
        self.non_zero_lengths = self.sconn_len[self.idx_x, self.idx_y]  # int64
        self.path_nor, self.path_mis = [[[[] for y in range(N_regions)] for x in range(N_regions)]] * 2

        #### is there more efficient way to do this?  --- to be updated.......
        for x, y, v in zip(self.idx_x, self.idx_y, self.non_zero_lengths):
            self.path_nor[x][y], self.path_mis[x][y] = [[0 for k in range(v)]] * 2

        # record the trajectory
        self.path_nor_history = []
        self.path_mis_history = []

        # continuous path and path history
        self.path_nor_cont, self.path_mis_cont = [np.zeros((N_regions, N_regions), dtype=np.int)] * 2
        self.path_nor_cont_history, self.path_mis_cont_history = [np.empty((0, self.N_regions, self.N_regions))] * 2

        # time step
        self.dt = dt

        # synthesis rate and clearance rate
        self.synthesis_rate = norm.cdf(snca) * self.dt
        self.clearance_rate = 1 - np.exp(-norm.cdf(gba) * self.dt)

        # probability of exit a path is set to v/sconn_len
        sconn_len[sconn_len == 0] = np.inf
        self.prob_exit = self.dt * v / sconn_len
        self.prob_exit[sconn_len == 0] = 0  # remove NaNs....

        # travel weights
        self.weights = np.exp(fcscale * fconn) * sconn_den

        self.weights = np.sum(self.weights, axis=1) * np.eye(self.N_regions) + self.weights

        # scale
        self.weights = self.weights / np.sum(self.weights, axis=1).reshape(self.N_regions, 1)

        self.weights_euler = self.weights * self.dt
        self.weights_euler[np.eye(self.N_regions) == 1] = 0
        self.weights_euler[np.eye(self.N_regions) == 1] = 1 - np.sum(self.weights_euler, axis=1)

        # region size
        self.roi_size = roi_size.flatten()

        self.synthesis_control = np.int_(roi_size.flatten())

    def normal_growth_region(self):
        """step: normal alpha-syn synthesized and cleared in regions"""

        self.nor -= np.array([np.sum(np.random.uniform(0, 1, (v,)) < k)
                              for k, v in zip(self.clearance_rate, self.nor)])

        ## synthesis
        self.nor += np.array([np.sum(np.random.uniform(0, 1, (v,)) < k)
                              for k, v in zip(self.synthesis_rate, self.synthesis_control)])
        # or self.roi_size)])

    def normal_growth_edge_discrete(self):
        """proteins are moving discretely in edges"""
        # alpha syn  -- from region to path
        # exit region
        exit_process = np.array([np.random.multinomial(self.nor[k], self.weights[k])
                                 for k in range(self.N_regions)], dtype=np.int)

        # alpha syn -- from path to region
        # enter region
        enter_process = np.zeros((self.N_regions, self.N_regions), dtype=np.int)

        for x, y in zip(self.idx_x, self.idx_y):
            # fetch then remove the last element
            enter_process[x, y] = self.path_nor[x][y].pop()
            # update paths
            self.path_nor[x][y].insert(0, exit_process[x, y])

        # update regions
        self.nor = exit_process[np.nonzero(np.eye(self.N_regions))] + np.sum(enter_process, axis=0)

    def normal_growth_edge(self):
        """proteins are moving contiously in edges"""

        # exit regions:
        exit_process = np.array([np.random.multinomial(self.nor[k], self.weights_euler[k])
                                 for k in range(self.N_regions)], dtype=np.int)
        exit_process[np.eye(self.N_regions) == 1] = 0  # remove diag

        # enter regions:
        enter_process = np.zeros((self.N_regions, self.N_regions), dtype=np.int)

        for x, y in zip(self.idx_x, self.idx_y):
            enter_process[x, y] = np.sum(
                np.random.uniform(0, 1, (self.path_nor_cont[x, y],)) < self.prob_exit[x, y] * self.dt)

        # update:
        self.path_nor_cont += (exit_process - enter_process)
        self.nor += (np.sum(enter_process, axis=0) - np.sum(exit_process, axis=1))

    def inject_mis(self, seed=41, initial_number=1):

        """inject initual_number misfolded protein into seed region"""
        # initial_number must be an interger
        self.mis[seed] = initial_number
        # print('inject %d misfolded alpha-syn into region %d' % (initial_number, seed))

    def misfolded_spread_edge_discrete(self):
        """ step in paths for normal and misfolded alpha syn"""

        ############## misfolded alpha synuclein ###########
        # exit regions
        exit_process = np.array([np.random.multinomial(v, self.weights[k])
                                 for k, v in enumerate(self.mis)], dtype=np.int)
        # alpha syn -- from path to region
        # enter region
        enter_process = np.zeros((self.N_regions, self.N_regions), dtype=np.int)
        for x, y in zip(self.idx_x, self.idx_y):
            # fetch then remove the last element
            enter_process[x, y] = self.path_mis[x][y].pop()
            # update paths
            self.path_mis[x][y].insert(0, exit_process[x, y])

        # update regions
        self.mis = exit_process[np.nonzero(np.eye(self.N_regions))] + np.sum(enter_process, axis=0)

        ########### for the normal alpha syuclein ###########
        exit_process = np.array([np.random.multinomial(self.nor[k], self.weights[k])
                                 for k in range(self.N_regions)], dtype=np.int)

        # alpha syn -- from path to region
        # enter region
        enter_process = np.zeros((self.N_regions, self.N_regions), dtype=np.int)

        for x, y in zip(self.idx_x, self.idx_y):
            # fetch then remove the last element
            enter_process[x, y] = self.path_nor[x][y].pop()
            # update paths
            self.path_nor[x][y].insert(0, exit_process[x, y])

        # update regions
        self.nor = exit_process[np.nonzero(np.eye(self.N_regions))] + np.sum(enter_process, axis=0)

    def misfolded_spread_edge(self):
        """proteins are moving continously"""
        ##### misfolded alpha synuclein #####
        exit_process = np.array([np.random.multinomial(v, self.weights_euler[k])
                                 for k, v in enumerate(self.mis)], dtype=np.int)
        exit_process[np.eye(self.N_regions) == 1] = 0

        enter_process = np.zeros((self.N_regions, self.N_regions), dtype=np.int)
        # enter regions:
        for x, y in zip(self.idx_x, self.idx_y):
            enter_process[x, y] = np.sum(
                np.random.uniform(0, 1, (self.path_mis_cont[x, y],)) < self.prob_exit[x, y] * self.dt)

        # update
        self.path_mis_cont += (exit_process - enter_process)
        self.mis += (np.sum(enter_process, axis=0) - np.sum(exit_process, axis=1))

        ####### normal alpha synuclein #######
        # exit regions:
        exit_process = np.array([np.random.multinomial(self.nor[k], self.weights_euler[k])
                                 for k in range(self.N_regions)], dtype=np.int) * self.dt
        exit_process[np.eye(self.N_regions) == 1] = 0

        # enter regions:
        enter_process = np.zeros((self.N_regions, self.N_regions), dtype=np.int)
        for x, y in zip(self.idx_x, self.idx_y):
            enter_process[x, y] = np.sum(
                np.random.uniform(0, 1, (self.path_nor_cont[x, y],)) < self.prob_exit[x, y] * self.dt)

        # update:
        self.path_nor_cont += (exit_process - enter_process)
        self.nor += (np.sum(enter_process, axis=0) - np.sum(exit_process, axis=1))

    def misfolded_spread_region(self, trans_rate=1):
        """clearance and synthesis of normal/misfolded alpha-syn/ transsmssion process in regions"""
        ## clearance
        cleared_nor = np.array([np.sum(np.random.uniform(0, 1, (v,)) < k)
                                for k, v in zip(self.clearance_rate, self.nor)])

        cleared_mis = np.array([np.sum(np.random.uniform(0, 1, (v,)) < k)
                                for k, v in zip(self.clearance_rate, self.mis)])

        self.prob_infected = 1 - np.exp(- (self.dt * self.mis * trans_rate / (self.roi_size)))
        # the remaining after clearance
        self.nor -= cleared_nor
        self.mis -= cleared_mis
        # self.prob_infected = 1 - np.exp(- (self.mis / self.roi_size) )
        infected_nor = np.array([np.sum(np.random.uniform(0, 1, (v,)) < k)
                                 for k, v in zip(self.prob_infected, self.nor)])
        # update self.nor and self.mis
        self.nor += (np.array([np.sum(np.random.uniform(0, 1, (v,)) < k)
                               for k, v in zip(self.synthesis_rate, self.roi_size)]) - infected_nor)
        self.mis += infected_nor

        # print(self.mis)

    def transmission_path(self, trans_rate_path):

        """transmission process in path (default shut down)"""
        for x, y, v in zip(self.idx_x, self.idx_y, self.non_zero_lengths):
            ### perhaps trans_rate_path should be set to 1/v ?
            # transmission rate is scaled by exp(distance) in voxel space
            path_nor_temp = np.array(self.path_nor[x][y])
            path_mis_temp = np.array(self.path_mis[x][y])
            rate_get_infected = (path_mis_temp * trans_rate_path) / np.exp(
                np.absolute(np.arange(v) - np.arange(v)[np.newaxis].T))
            prob_get_infected = 1 - np.exp(np.sum(-rate_get_infected, axis=1))
            infected_path = np.array([np.sum(np.random.uniform(0, 1, (k,)) < v) for
                                      k, v in zip(self.path_nor[x][y], prob_get_infected)])

            # update self.path_nor and self.path_mis
            path_nor_temp -= infected_path
            path_mis_temp += infected_path

            self.path_nor[x][y] = path_nor_temp.tolist()
            self.path_mis[x][y] = path_mis_temp.tolist()

    def record_to_history_discrete(self):
        """record the results of each step into the recorder"""
        self.nor_history = np.append(self.nor_history, self.nor[np.newaxis], axis=0)
        self.mis_history = np.append(self.mis_history, self.mis[np.newaxis], axis=0)

        # record the mobility patterns in edges
        # self.path_nor_history.append(self.path_nor)
        # self.path_mis_history.append(self.path_mis)

    def record_to_history(self):
        """record the results of each step"""
        self.nor_history = np.append(self.nor_history, self.nor[np.newaxis], axis=0)
        self.mis_history = np.append(self.mis_history, self.mis[np.newaxis], axis=0)

        # record the mobility patterns in edges
        # self.path_nor_cont_history = np.append(self.path_nor_cont_history, self.path_nor_cont.reshape(1, 42, 42), axis = 0)
        # self.path_mis_cont_history = np.append(self.path_mis_cont_history, self.path_mis_cont.reshape(1, 42, 42), axis = 0)


class SIR_model:
    """An SIR model to simulate the spread of alpha-syn"""

    # constructor
    def __init__(self, v=1, N_regions=42, dt=0.01, sconn_len=None, sconn_den=None, growth_rate=None, clearance_rate=None, region_size=None,
                 fconn=None, fcscale=None):
        # number of regions
        self.N_regions = N_regions
        self.dt = dt
        # store number of normal and misfolded proteins in regions
        self.nor, self.mis = [np.zeros((N_regions,))] * 2
        self.nor_history, self.mis_history = [np.empty((0, N_regions))] * 2

        # store number of normal and misfoded proteins in paths
        self.path_nor, self.path_mis = [np.zeros((N_regions, N_regions))] * 2
        self.path_nor_history, self.path_mis_history = [np.empty((0, N_regions, N_regions))] * 2

        # index of connected components
        (self.idx_x, self.idx_y) = np.nonzero(sconn_len)
        self.non_zero_lengths = sconn_len[np.nonzero(sconn_len)]

        # probability of exit a path is set to v/sconn_len
        sconn_len[sconn_len == 0] = np.inf
        self.prob_exit = v / sconn_len
        self.prob_exit[sconn_len == 0] = 0

        # synthesis rate and  clearance rate
        self.synthesis_rate = norm.cdf(snca) * self.dt
        self.clearance_rate = 1 - np.exp(-norm.cdf(gba) * self.dt)

        # mobility pattern multinomial distribution
        self.weights = np.exp(fcscale * fconn) * sconn_den
        self.weights = np.diag(np.sum(sconn_den, axis=1)) + sconn_den
        self.weights = self.weights / np.sum(self.weights, axis=1)[np.newaxis].T

        # region size
        self.roi_size = roi_size.flatten()

    def normal_growth_region(self):
        """normal alpha-syn growing"""
        self.nor += (self.roi_size * self.synthesis_rate - self.nor * self.clearance_rate)

    def normal_growth_edge(self):
        # enter paths
        enter_process = self.nor.reshape(self.N_regions, 1) * self.weights * self.dt
        enter_process[np.eye(self.N_regions) == 1] = 0  # remove diag

        # exit paths
        exit_process = self.path_nor * self.prob_exit * self.dt

        # update paths and regions
        self.nor += (np.sum(exit_process, axis=0) - np.sum(enter_process, axis=1))
        self.path_nor += (enter_process - exit_process)

    def inject_mis(self, seed=41, initial_number=1):
        """inject misfolded alpha-syn in seed region"""

        self.mis[seed] = initial_number
        # print('inject %d misfolded alpha-syn into region %d' % (initial_number, seed))

    def misfolded_spread_edge(self):
        ##### misfolded alpha-syn
        # enter paths
        enter_process = self.mis.reshape(self.N_regions, 1) * self.weights * self.dt
        enter_process[np.eye(self.N_regions) == 1] = 0  # remove diag

        # exit paths
        exit_process = self.path_mis * self.prob_exit

        # update paths and regions
        self.mis += (-np.sum(enter_process, axis=1) + np.sum(exit_process, axis=0))
        self.path_mis += (enter_process - exit_process)

        ###### normal alpha-syn
        # enter paths
        enter_process = self.nor.reshape(self.N_regions, 1) * self.weights * self.dt
        enter_process[np.eye(self.N_regions) == 1] = 0  # remove diag

        # exit paths
        exit_process = self.path_nor * self.prob_exit * self.dt

        # update paths and regions
        self.nor += (np.sum(exit_process, axis=0) - np.sum(enter_process, axis=1))
        self.path_nor += (enter_process - exit_process)

    def misfolded_spread_region(self, trans_rate=1):
        '"""the transmission process inside regions"""'

        prob_get_infected = 1 - np.exp(-self.mis * trans_rate * self.dt / self.roi_size)
        # clear process
        self.nor -= self.nor * self.clearance_rate
        self.mis -= self.mis * self.clearance_rate

        infected = self.nor * prob_get_infected
        self.nor += (self.roi_size * self.synthesis_rate - infected)
        self.mis += (infected)

    def transmission_path(self):
        """the transmission process in paths"""
        pass

    def record_to_history(self):
        """record the results"""
        self.nor_history = np.append(self.nor_history, self.nor[np.newaxis], axis=0)
        self.mis_history = np.append(self.mis_history, self.mis[np.newaxis], axis=0)

        self.path_nor_history = np.append(self.path_nor_history,
                                          self.path_nor.reshape(1, self.N_regions, self.N_regions), axis=0)
        self.path_mis_history = np.append(self.path_mis_history,
                                          self.path_mis.reshape(1, self.N_regions, self.N_regions), axis=0)


class PopulationModel:
    """An SIR model to simulate the spread of alpha-syn"""

    # constructor
    def __init__(self, v=1, N_regions=42, dt=0.01, sconn_len=None, sconn_den=None, growth_rate=None, clearance_rate=None, region_size=None,
                 fconn=None, fcscale=None):
        # number of regions
        self.N_regions = N_regions
        self.dt = dt
        # store number of normal and misfolded proteins in regions
        self.nor, self.mis = [np.zeros((N_regions,))] * 2
        self.nor_history, self.mis_history = [np.empty((0, N_regions))] * 2

        # store number of normal and misfoded proteins in paths
        self.path_nor, self.path_mis = [np.zeros((N_regions, N_regions))] * 2
        self.path_nor_history, self.path_mis_history = [np.empty((0, N_regions, N_regions))] * 2

        # index of connected components
        (self.idx_x, self.idx_y) = np.nonzero(sconn_len)
        self.non_zero_lengths = sconn_len[np.nonzero(sconn_len)]

        # probability of exit a path is set to v/sconn_len
        sconn_len[sconn_len == 0] = np.inf
        self.prob_exit = v / sconn_len
        self.prob_exit[sconn_len == 0] = 0

        # synthesis rate and  clearance rate
        self.synthesis_rate = norm.cdf(snca) * self.dt
        self.clearance_rate = 1 - np.exp(-norm.cdf(gba) * self.dt)

        # mobility pattern multinomial distribution
        self.weights = np.exp(fcscale * fconn) * sconn_den
        self.weights = np.diag(np.sum(sconn_den, axis=1)) + sconn_den
        self.weights = self.weights / np.sum(self.weights, axis=1)[np.newaxis].T

        # region size
        self.roi_size = roi_size.flatten()

    def normal_growth_region(self):
        """normal alpha-syn growing"""
        self.nor += (self.roi_size * self.synthesis_rate - self.nor * self.clearance_rate)

    def normal_growth_edge(self):
        # enter paths
        enter_process = self.nor.reshape(self.N_regions, 1) * self.weights * self.dt
        enter_process[np.eye(self.N_regions) == 1] = 0  # remove diag

        # exit paths
        exit_process = self.path_nor * self.prob_exit * self.dt

        # update paths and regions
        self.nor += (np.sum(exit_process, axis=0) - np.sum(enter_process, axis=1))
        self.path_nor += (enter_process - exit_process)

    def inject_mis(self, seed=41, initial_number=1):
        """inject misfolded alpha-syn in seed region"""

        self.mis[seed] = initial_number
        # print('inject %d misfolded alpha-syn into region %d' % (initial_number, seed))

    def misfolded_spread_edge(self):
        ##### misfolded alpha-syn
        # enter paths
        enter_process = self.mis.reshape(self.N_regions, 1) * self.weights * self.dt
        enter_process[np.eye(self.N_regions) == 1] = 0  # remove diag

        # exit paths
        exit_process = self.path_mis * self.prob_exit

        # update paths and regions
        self.mis += (-np.sum(enter_process, axis=1) + np.sum(exit_process, axis=0))
        self.path_mis += (enter_process - exit_process)

        ###### normal alpha-syn
        # enter paths
        enter_process = self.nor.reshape(self.N_regions, 1) * self.weights * self.dt
        enter_process[np.eye(self.N_regions) == 1] = 0  # remove diag

        # exit paths
        exit_process = self.path_nor * self.prob_exit * self.dt

        # update paths and regions
        self.nor += (np.sum(exit_process, axis=0) - np.sum(enter_process, axis=1))
        self.path_nor += (enter_process - exit_process)

    def misfolded_spread_region(self, trans_rate=1):
        '"""the transmission process inside regions"""'

        prob_get_infected = 1 - np.exp(-self.mis * trans_rate * self.dt / self.roi_size)
        # clear process
        self.nor -= self.nor * self.clearance_rate
        self.mis -= self.mis * self.clearance_rate

        infected = self.nor * prob_get_infected
        self.nor += (self.roi_size * self.synthesis_rate - infected)
        self.mis += (infected)

    def transmission_path(self):
        """the transmission process in paths"""
        pass

    def record_to_history(self):
        """record the results"""
        self.nor_history = np.append(self.nor_history, self.nor[np.newaxis], axis=0)
        self.mis_history = np.append(self.mis_history, self.mis[np.newaxis], axis=0)

        self.path_nor_history = np.append(self.path_nor_history,
                                          self.path_nor.reshape(1, self.N_regions, self.N_regions), axis=0)
        self.path_mis_history = np.append(self.path_mis_history,
                                          self.path_mis.reshape(1, self.N_regions, self.N_regions), axis=0)