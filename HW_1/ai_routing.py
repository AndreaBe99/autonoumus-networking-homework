from collections import defaultdict

import numpy as np

from src.routing_algorithms.BASE_routing import BASE_routing
from src.routing_algorithms.georouting import GeoRouting
import random
import src.utilities.utilities as util


class AIPath(BASE_routing):
    exploration = 0
    exploitation = 0

    def __init__(self, drone, simulator):
        BASE_routing.__init__(self, drone, simulator)
        # random generator
        self.rnd_for_routing_ai = np.random.RandomState(self.simulator.seed)
        self.taken_actions = {}  # id event : (old_action)
        self.epsilon = 0.05
        self.n = [0, 0]
        self.q_value = [0, 0]
        self.dictionary = defaultdict(list)  # {id_event : [drones already rewarded]}
        self.expired_packets = defaultdict(list)  # {id_event : [drones that expired packet id_event]}


    def feedback(self, drone, id_event, delay, outcome):
        """ return a possible feedback, if the destination drone has received the packet """

        if id_event not in self.taken_actions:
            return None
		
		# reward_m is a Multiplier for the reward
        action, reward_m = self.taken_actions[id_event]
        reward = (self.simulator.event_duration - delay) * reward_m
        self.n[action] += 1
        self.q_value[action] = self.q_value[action] + ( (reward - self.q_value[action]) / self.n[action] )

    def relay_selection(self, opt_neighbors, pkd):
        # now epsilon greedy selection of the action
        # 1) case epsilon, we take a random action
        if random.uniform(0, 1) < self.epsilon:
            action = random.randint(0, 1)
        # 2) case of 1 - epsilon we take the greatest q-value
        else:
            action = np.argmax(self.q_value) # with funcion argmax of numpy we take the max of the q_value array

        drone_to_send = None

        # exploration keep the packet
        if action == 1:
            AIPath.exploitation += 1
            drone_to_send =  None
        # exploitation give the packet with GEO routing and check if packets expired from the drone choose
        else:
            AIPath.exploration += 1
            drone_to_send = GeoRouting.relay_selection(self, opt_neighbors, pkd)

        if drone_to_send != None:
            # If the recipient is a ferry
            if drone_to_send.identifier <= 2:
                # If it's headed to the depot
                if len(drone_to_send.waypoint_history) > 0 and drone_to_send.waypoint_history[-1] != self.simulator.depot_coordinates:
                    reward_m= 10
                else:
                    reward_m = 8

            # If the recipient is a routing drone
            elif drone_to_send.identifier > 2:             
                #If a drone ferry sends a packet to a routing drone
                if self.drone.identifier <= 2: 
                    #If it's headed to the depot
                    if len(drone_to_send.waypoint_history) > 0 and drone_to_send.waypoint_history[-1] == drone_to_send.path[-2]:
                        reward_m = 10
                    else:
                        reward_m =  1
                # If a routing drone sends a packet to another routing drone
                elif self.drone.identifier > 2: 
                    # If it's headed to the depot
                    if len(drone_to_send.waypoint_history) > 0 and drone_to_send.waypoint_history[-1] == drone_to_send.path[-2]:
                        reward_m = 10
                    # The recipient drone must return to the depot first
                    if len(self.drone.path) - len(self.drone.waypoint_history) > len(drone_to_send.path) - len(drone_to_send.waypoint_history):
                        reward_m =  5
                    else:
                        reward_m= 1 
        else:
            n_step_sender = 0
            # We calculate the number of steps required to reach the depot for the drone that held the package
            if self.drone.identifier > 2:
                n_step_sender = self.n_step(self.drone)
            # We calculate the number of steps for the recipients
            min_n_step = float('inf')
            for pkd_id, drone in opt_neighbors:
                n_step_recipient = self.n_step(drone)
                if n_step_recipient < min_n_step:
                    min_n_step = n_step_recipient
   
            if n_step_sender > min_n_step:
                reward_m = 10
            else:
                reward_m = 1

        self.taken_actions[pkd.event_ref.identifier] = (action, reward_m)

        return drone_to_send



    def n_step(self, drone):
        # IF I AM A ROUTING DRONE
        if drone.identifier > 2:
            # We calculate the number of steps required to reach the depot
            # If it is the first execution of the path
            if len(drone.waypoint_history) <= len(drone.waypoint_history):
                indx = len(drone.waypoint_history)
                n_step = len(drone.path) - indx
            # If it's not the first round of the path
            else:
                n_iter = len(drone.waypoint_history) % len(drone.path) # MODULO
                indx = int(len(drone.waypoint_history)/n_iter)
                n_step = len(drone.path) - indx
        # IF I AM A FERRIES
        else:
            n_step = 1
            # It is heading for the depot
            if drone.waypoint_history[-1] != self.simulator.depot_coordinates:
                n_step = 0

        return n_step


    def print(self):
        """
            This method is called at the end of the simulation, can be usefull to print some
                metrics about the learning process
        """
        print("Which times we got exploration and exploitation?")
        print("Exploration  -> ", AIPath.exploration)
        print("Exploitation -> ", AIPath.exploitation)
