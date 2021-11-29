import random
from collections import defaultdict

import numpy as np
from src.routing_algorithms.BASE_routing import BASE_routing
from src.utilities import utilities as util, config


# We got three types of actions:
# 0) Send packet to other drones
# 1) Wait
# 2) Go physically to the depot

class OLDAIRouting(BASE_routing):
    send_pkt = 0
    keep_pkt = 0
    move_to_depot = 0

    def __init__(self, drone, simulator):
        BASE_routing.__init__(self, drone, simulator)
        # random generator
        self.rnd_for_routing_ai = np.random.RandomState(self.simulator.seed)
        self.taken_actions = {}  # id event : (old_state, old_action)
        self.epsilon = 0.02
        # we got three actions now but we apply RL only in the first two actions
        self.n = [0, 0]
        self.q_value = [0, 0]
        self.dictionary = defaultdict(list)  # {id_event : [drones already rewarded]}
        self.expired_packets = defaultdict(list)  # {id_event : [drones that expired packet id_event]}

    def feedback(self, drone, id_event, delay, outcome):
        """ return a possible feedback, if the destination drone has received the packet """
        # already feedbacked drones
        if drone in self.dictionary[id_event]:
            return None
        if config.DEBUG:
            # Packets that we delivered and still need a feedback
            print("Drone: ", self.drone.identifier, "---------- has delivered: ", self.taken_actions)

            # outcome == -1 if the packet/event expired; 0 if the packets has been delivered to the depot
            # Feedback from a delivered or expired packet
            print("Drone: ", self.drone.identifier, "---------- just received a feedback:",
                  "Drone:", drone, " - id-event:", id_event, " - delay:", delay, " - outcome:", outcome)
        if outcome == -1:
            self.dictionary[id_event].append(drone)
            if drone not in self.expired_packets[id_event]:
                self.expired_packets[id_event].append(drone)
        else:
            self.dictionary[id_event].append(drone)
            if drone in self.expired_packets[id_event]:
                self.expired_packets[id_event].remove(drone)
        reward = self.simulator.event_duration - delay

        if id_event not in self.taken_actions:
            return None

        action = self.taken_actions[id_event]
        # If we go to the depot we don't update q_values
        if action == 2:
            return None
        self.n[action] += 1

        self.q_value[action] = self.q_value[action] + ((1 / self.n[action]) * (reward - self.q_value[action]))

    def relay_selection(self, opt_neighbors, pkd):
        # Before the epsilon greedy selection we check if the action 2 can be good
        if util.euclidean_distance(self.simulator.depot_coordinates,
                                   self.drone.coords) < self.drone.depot.communication_range * 1.1:
            OLDAIRouting.move_to_depot += 1
            self.taken_actions[pkd.event_ref] = 2
            return -1
        # now epsilon greedy selection of the action
        # case epsilon which takes random action
        if random.uniform(0, 1) < self.epsilon:
            action = random.randint(0, 1)
        # case of 1 - epsilon we take the greatest q-value
        else:
            action = np.argmax(self.q_value)
        # take the action
        self.taken_actions[pkd.event_ref.identifier] = action
        # exploration keep the packet
        if self.taken_actions[pkd.event_ref.identifier] == 1:
            OLDAIRouting.keep_pkt += 1
            reward = 0
            return None
        # exploitation give the packet with GEO routing and check if packets expired from the drone choose
        else:
            OLDAIRouting.send_pkt += 1
            reward = 10
            drone_to_send = self.ModGeoRouting(opt_neighbors)
        self.n[action] += 1
        self.q_value[action] = self.q_value[action] + ((1 / self.n[action]) * (reward - self.q_value[action]))
        return drone_to_send

    def print(self):
        """
            This method is called at the end of the simulation, can be usefull to print some
                metrics about the learning process
        """
        print("\n############## PRINT ###############")
        print("Number of Drone: ", self.simulator.n_drones)
        print("Send the Packet: ", OLDAIRouting.send_pkt)
        print("Keep the Packet: ", OLDAIRouting.keep_pkt)
        print("Move to Depot: ", OLDAIRouting.move_to_depot)
        print("####################################\n")

    def ModGeoRouting(self, opt_neighbors):
        drone_to_send = None
        if util.euclidean_distance(self.drone.depot.coords,
                                   self.drone.coords) <= self.simulator.depot_com_range and self.drone not in self.expired_packets:
            return drone_to_send

        # GEO Routing with check if this drone is successful
        best_drone_distance_from_depot = util.euclidean_distance(self.simulator.depot.coords, self.drone.coords)
        for hpk, drone_istance in opt_neighbors:
            exp_position = hpk.cur_pos  # without estimation, a simple geographic approach
            exp_distance = util.euclidean_distance(exp_position, self.simulator.depot.coords)
            if not self.expired_packets:
                if exp_distance < best_drone_distance_from_depot:
                    best_drone_distance_from_depot = exp_distance
                    drone_to_send = drone_istance
            else:
                if exp_distance < best_drone_distance_from_depot and drone_istance not in self.expired_packets:
                    best_drone_distance_from_depot = exp_distance
                    drone_to_send = drone_istance

        return drone_to_send
