import numpy as np

from src.routing_algorithms.georouting_w_move import GeoMoveRouting
from src.routing_algorithms.georouting import GeoRouting
from src.utilities import config
from src.utilities import utilities as util
from src.routing_algorithms.BASE_routing import BASE_routing
import random


class OPTQlearning(BASE_routing):
    keep_pkt = 0
    send_pkt = 0
    move_to_depot_1 = 0
    move_to_depot_2 = 0

    def __init__(self, drone, simulator):
        BASE_routing.__init__(self, drone, simulator)
        # random generator
        self.rnd_for_routing_ai = np.random.RandomState(self.simulator.seed)
        self.taken_actions = {}  # id event : (old_action)

        self.cell_number = pow(int(self.simulator.env_width / self.simulator.prob_size_cell), 2)
        self.action_number = 4  # we consider 4 actions: 0:send_pkt, 1:keep_pkt, 2:move_to_depot --> 1, 3:move_to_depot --> 2
        self.q_value = [[10 for i in range(self.action_number)] for j in range(self.cell_number)]  # [N-cells][N-action]
        self.epsilon = 0
        self.alpha = 0.7
        self.gamma = 0.6
        self.to_depot = False

    def feedback(self, drone, id_event, delay, outcome, depot_index=None):
        """ return a possible feedback, if the destination drone has received the packet """
        if config.DEBUG:
            # Packets that we delivered and still need a feedback
            print("Drone: ", self.drone.identifier, "---------- has delivered: ", self.taken_actions)

            # outcome == -1 if the packet/event expired; 0 if the packets has been delivered to the depot
            # Feedback from a delivered or expired packet
            print("Drone: ", self.drone.identifier, "---------- just received a feedback:",
                  "Drone:", drone, " - id-event:", id_event, " - delay:", delay, " - outcome:", outcome, " - to depot: "
                  , depot_index)

        if id_event in self.taken_actions:
            action, old_cell, next_target_cell, mul_reward, time_to_depot = self.taken_actions[id_event]

            if action == 2 or action == 3:
                reward = time_to_depot / 10 * mul_reward
            else:
                reward = mul_reward
            self.to_depot = False
            self.q_value[old_cell][action] = self.q_value[old_cell][action] + self.alpha * (
                    reward + self.gamma * max(self.q_value[next_target_cell]) - self.q_value[old_cell][action])

    def relay_selection(self, opt_neighbors, pkd):
        """ arg min score  -> geographical approach, take the drone closest to the depot """

        cell_index = int(util.TraversedCells.coord_to_cell(size_cell=self.simulator.prob_size_cell,
                                                           width_area=self.simulator.env_width,
                                                           x_pos=self.drone.coords[0], y_pos=self.drone.coords[1])[0])
        action = None

        # now epsilon greedy selection of the action
        # 1) case epsilon, we take a random action
        if random.uniform(0, 1) < self.epsilon:
            action = random.randint(0, 3)
        # 2) case of 1 - epsilon we take the greatest q-value
        else:
            action = np.argmax(self.q_value[cell_index])

        # With self.drone.next_target() we know the next cell, i.e. the next state
        next_target_coord = self.drone.next_target()
        next_target_cell = None
        drone_to_send = None

        # -2 --> move to depot
        if action == 3:
            OPTQlearning.move_to_depot_2 += 1
            drone_to_send = -2
            # Cell of depot
            next_target_cell = int(util.TraversedCells.coord_to_cell(size_cell=self.simulator.prob_size_cell,
                                                                     width_area=self.simulator.env_width,
                                                                     x_pos=self.simulator.depot_coordinates[1][0],
                                                                     y_pos=self.simulator.depot_coordinates[1][1])[0])

        # -1 --> move to depot
        if action == 2:
            OPTQlearning.move_to_depot_1 += 1
            drone_to_send = -1
            # Cell of depot
            next_target_cell = int(util.TraversedCells.coord_to_cell(size_cell=self.simulator.prob_size_cell,
                                                                     width_area=self.simulator.env_width,
                                                                     x_pos=self.simulator.depot_coordinates[0][0],
                                                                     y_pos=self.simulator.depot_coordinates[0][1])[0])

        # None --> no transmission, keep the packet
        if action == 1:
            OPTQlearning.keep_pkt += 1
            drone_to_send = None
            next_target_cell = int(util.TraversedCells.coord_to_cell(size_cell=self.simulator.prob_size_cell,
                                                                     width_area=self.simulator.env_width,
                                                                     x_pos=next_target_coord[0],
                                                                     y_pos=next_target_coord[1])[0])

        # send packet
        elif action == 0:
            OPTQlearning.send_pkt += 1
            drone_to_send = GeoRouting.relay_selection(self, opt_neighbors, pkd)
            # If there are no neighbors
            if drone_to_send is None:
                action = 1
                OPTQlearning.send_pkt -= 1
                OPTQlearning.keep_pkt += 1
            next_target_cell = int(util.TraversedCells.coord_to_cell(size_cell=self.simulator.prob_size_cell,
                                                                     width_area=self.simulator.env_width,
                                                                     x_pos=next_target_coord[0],
                                                                     y_pos=next_target_coord[1])[0])

        mul_reward = self.calculate_reward(opt_neighbors, drone_to_send, cell_index, action)

        self.q_value[cell_index][action] = self.q_value[cell_index][action] + self.alpha * (
                mul_reward + self.gamma * max(self.q_value[next_target_cell]) - self.q_value[cell_index][action])

        time_to_depot = 0
        # In order not to override action 2 on the way to the depot
        if action == 2 or action == 3:
            distance_from_depot = util.euclidean_distance(self.simulator.depot_coordinates[action - 2],
                                                          self.drone.coords)
            time_to_depot = distance_from_depot / self.drone.speed
            self.taken_actions[
                pkd.event_ref.identifier] = action, cell_index, next_target_cell, mul_reward, time_to_depot
            self.to_depot = True

        if self.to_depot == False:
            self.taken_actions[
                pkd.event_ref.identifier] = action, cell_index, next_target_cell, mul_reward, time_to_depot

        return drone_to_send

    def calculate_reward(self, opt_neighbors, drone_to_send, cell_index, action):
        # We search if there is a drone that goes to the depot
        drones_to_depot = []
        for hpk, drone_instance in opt_neighbors:
            if drone_instance.next_target() == self.simulator.depot_coordinates[0] or drone_instance.next_target() == \
                    self.simulator.depot_coordinates[1]:
                drones_to_depot.append(drone_instance)

        # We calculate the multiplier for the reward
        mul_reward = 0
        # Number of cells in a row
        num_cell_in_row = int(self.simulator.env_width / self.simulator.prob_size_cell)

        depot_1_cell = int(util.TraversedCells.coord_to_cell(size_cell=self.simulator.prob_size_cell,
                                                             width_area=self.simulator.env_width,
                                                             x_pos=self.simulator.depot_coordinates[0][0],
                                                             y_pos=self.simulator.depot_coordinates[0][1])[0])
        depot_2_cell = int(util.TraversedCells.coord_to_cell(size_cell=self.simulator.prob_size_cell,
                                                             width_area=self.simulator.env_width,
                                                             x_pos=self.simulator.depot_coordinates[1][0],
                                                             y_pos=self.simulator.depot_coordinates[1][1])[0])
        # Cycle on the number of rows / columns
        for row_number in range(num_cell_in_row):
            # the row where the drone is located
            if cell_index < (row_number + 1) * num_cell_in_row:
                if action == 0:
                    # If the intended drone is directed to the depot I will give a high reward
                    if drone_to_send in drones_to_depot:
                        mul_reward = 5
                    else:
                        # If the drone is in a cell near the depot, the reward will be lower
                        # We calculate the reward based on the distance from the cell to the depot
                        if row_number < int(num_cell_in_row / 2):
                            mul_reward = (row_number + 1) * abs(
                                cell_index - (depot_1_cell + num_cell_in_row * row_number))
                        else:
                            mul_reward = (num_cell_in_row - row_number) * abs(
                                cell_index - (depot_2_cell - (num_cell_in_row * (num_cell_in_row - row_number - 1))))
                        # The reward is affected by the number of drones, in fact, with a greater number of drones there
                        # will be a greater number of events at the depot
                        mul_reward = mul_reward * self.simulator.n_drones
                        break
                elif action == 1:
                    # If I had a neighbor who already went to the depot I will give a very low reward
                    if len(drones_to_depot) != 0:
                        mul_reward = -5
                    else:
                        # I calculate the reward based on the distance from the cell to the depot
                        # If the drone is in a cell near the depot, the reward will be higher
                        if row_number < int(num_cell_in_row / 2):
                            mul_reward = (num_cell_in_row - row_number) * abs(
                                cell_index - (depot_1_cell + num_cell_in_row * row_number))
                        else:
                            mul_reward = (row_number + 1) * abs(
                                cell_index - (depot_2_cell - (num_cell_in_row * (num_cell_in_row - row_number - 1))))

                        # Forse Questo è giusto
                        # mul_reward = abs(num_cell_in_row / 2 - row_number + 1) * abs((num_cell_in_row / 2) * (row_number + 1) - cell_index)
                        break
                elif action == 2:
                    # If I had a neighbor who already went to the depot I will give a very low reward
                    if len(drones_to_depot) != 0:
                        mul_reward = -5
                    else:
                        # We calculate the reward based on the distance from the cell to the depot
                        # If the drone is in a cell near the depot, the reward will be higher
                        # because the drone wastes less energy to return to the mission
                        mul_reward = - (row_number + 1) * abs(
                            cell_index - (depot_1_cell + num_cell_in_row * row_number))
                        # The reward is affected by the number of drones, in fact, with a greater number of drones there
                        # will be a greater number of events at the depot
                        mul_reward = mul_reward * self.simulator.n_drones
                        break
                elif action == 3:
                    # If I had a neighbor who already went to the depot I will give a very low reward
                    if len(drones_to_depot) != 0:
                        mul_reward = -5
                    else:
                        # We calculate the reward based on the distance from the cell to the depot
                        # If the drone is in a cell near the depot, the reward will be higher
                        # because the drone wastes less energy to return to the mission
                        mul_reward = - (num_cell_in_row - row_number) * abs(
                            cell_index - (depot_2_cell - (num_cell_in_row * (num_cell_in_row - row_number - 1))))
                        # The reward is affected by the number of drones, in fact, with a greater number of drones there
                        # will be a greater number of events at the depot
                        mul_reward = mul_reward * self.simulator.n_drones
                        break

        return mul_reward

    def print(self):
        """
            This method is called at the end of the simulation, can be usefull to print some
                metrics about the learning process
        """
        print("\n############## PRINT ###############")
        print("Number of Drone: ", self.simulator.n_drones)
        print("Send the Packet: ", OPTQlearning.send_pkt)
        print("Keep the Packet: ", OPTQlearning.keep_pkt)
        print("Move to Depot 1: ", OPTQlearning.move_to_depot_1)
        print("Move to Depot 2: ", OPTQlearning.move_to_depot_2)
        print("####################################\n")
