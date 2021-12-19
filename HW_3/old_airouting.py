from collections import defaultdict

import numpy as np

from src.utilities import config
from src.utilities import utilities as util
from src.routing_algorithms.BASE_routing import BASE_routing
import random


class OLDAIRouting(BASE_routing):
    keep_pkt = 0
    send_pkt = 0
    move_to_depot_1 = 0
    move_to_depot_2 = 0

    def __init__(self, drone, simulator):
        BASE_routing.__init__(self, drone, simulator)
        # random generator
        self.rnd_for_routing_ai = np.random.RandomState(self.simulator.seed)
        self.taken_actions = {}  # id event : (old_action)
        self.n = [0, 0, 0, 0]  # 4 actions
        self.q_value = [0, 0, 0, 0]  # 4 actions
        self.epsilon = 0.02
        self.to_depot = False
        self.dictionary = defaultdict(list)
        self.expired_packets = defaultdict(list)

    def feedback(self, drone, id_event, delay, outcome, depot_index=None):
        """ return a possible feedback, if the destination drone has received the packet """
        if drone in self.dictionary[id_event]:
            return None
        if config.DEBUG:
            # Packets that we delivered and still need a feedback
            print("Drone: ", self.drone.identifier, "---------- has delivered: ", self.taken_actions)

            # outcome == -1 if the packet/event expired; 0 if the packets has been delivered to the depot
            # Feedback from a delivered or expired packet
            print("Drone: ", self.drone.identifier, "---------- just received a feedback:",
                  "Drone:", drone, " - id-event:", id_event, " - delay:", delay, " - outcome:", outcome, " - to depot: "
                  , depot_index)
        if outcome == -1:
            self.dictionary[id_event].append(drone)
            if drone not in self.expired_packets[id_event]:
                self.expired_packets[id_event].append(drone)
        reward = self.simulator.event_duration - delay
        if id_event not in self.taken_actions:
            return None

        action = self.taken_actions[id_event]
        # If we go to the depot we don't update q_values
        if action == 2 or action == 3:
            reward = reward / 10  # penalize the action to go physically to the depot
        self.n[action] += 1
        self.q_value[action] = self.q_value[action] + ((1 / self.n[action]) * (reward - self.q_value[action]))

    def relay_selection(self, opt_neighbors, pkd):
        """ arg min score  -> geographical approach, take the drone closest to the depot """
        # now epsilon greedy selection of the action
        drone_to_send = None
        reward = 0
        # case epsilon which takes random action
        if random.uniform(0, 1) < self.epsilon:
            action = random.randint(0, 3)
        # case of 1 - epsilon we take the greatest q-value
        else:
            action = np.argmax(self.q_value)
        # take the action
        self.taken_actions[pkd.event_ref.identifier] = action
        # exploration keep the packet
        if self.taken_actions[pkd.event_ref.identifier] == 1:
            OLDAIRouting.keep_pkt += 1
            reward = 0
            drone_to_send = None
        elif self.taken_actions[pkd.event_ref.identifier] == 2:
            OLDAIRouting.move_to_depot_1 += 1
            reward = - 10
            drone_to_send = -1
        elif self.taken_actions[pkd.event_ref.identifier] == 3:
            OLDAIRouting.move_to_depot_2 += 1
            reward = - 10
            drone_to_send = - 2
        # exploitation give the packet with GEO routing and check if packets expired from the drone choose
        elif self.taken_actions[pkd.event_ref.identifier] == 0:
            OLDAIRouting.send_pkt += 1
            reward = 10
            drone_to_send = self.ModGeoRouting(opt_neighbors)
        self.n[action] += 1
        self.q_value[action] = self.q_value[action] + ((1 / self.n[action]) * (reward - self.q_value[action]))
        return drone_to_send

    def ModGeoRouting(self, opt_neighbors):
        drone_to_send = None
        if util.euclidean_distance(self.simulator.depot.list_of_coords[0],
                                   self.drone.coords) <= self.simulator.depot_com_range and self.drone not in self.expired_packets:
            return drone_to_send

        # GEO Routing with check if this drone is successful
        best_drone_distance_from_depot = util.euclidean_distance(self.simulator.depot.list_of_coords[0], self.drone.coords)
        for hpk, drone_istance in opt_neighbors:
            exp_position = hpk.cur_pos  # without estimation, a simple geographic approach
            exp_distance = util.euclidean_distance(exp_position, self.simulator.depot.list_of_coords[0])
            if not self.expired_packets:
                if exp_distance < best_drone_distance_from_depot:
                    best_drone_distance_from_depot = exp_distance
                    drone_to_send = drone_istance
            else:
                if exp_distance < best_drone_distance_from_depot and drone_istance not in self.expired_packets:
                    best_drone_distance_from_depot = exp_distance
                    drone_to_send = drone_istance

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
        print("Move to Depot 1: ", OLDAIRouting.move_to_depot_1)
        print("Move to Depot 2: ", OLDAIRouting.move_to_depot_2)
        print("####################################\n")
