import numpy as np

from src.routing_algorithms.georouting_w_move import GeoMoveRouting
from src.routing_algorithms.georouting import GeoRouting
from src.utilities import config
from src.utilities import utilities as util
from src.routing_algorithms.BASE_routing import BASE_routing
import random


class AIRouting(BASE_routing):
    keep_pkt = 0
    send_pkt = 0
    move_to_depot = 0

    """
    Il buon funzionamento dell'algoritmo credo dipenda molto dalla posizione in cui effettuo l'azione e che tipo di 
    azione. Per questo credo che le celle siano un ottimo modo per strutturare l'algoritmo.

    Ogni cella ha un array di 3 valori corrispondente alle azioni.
    - Azione 2:
        - Più sono lontano dal depot più l'azione 2  deve essere negativa.
        - Più sono vicino al depot più l'azione 2  deve essere positiva.

    - Azione 1 (keep_pkt):
        - Più sono lontano dal depot più l'azione 1 deve essere positiva.
        - Più sono vicino al depot più l'azione 2 deve essere negativa.
            Questo perchè se sono vicino conviene tentare di consegnare il pacchetto
        - Se tra i vicini ho un drone diretto al depot e scelgo l'azione 1 devo avere un reward molto negativo.

    - Azione 0 (send_pkt):
        - Invio esclusivamente a droni diretti verso il depot, altrimenti faccio l'azione 1, questo perchè
            inviare pacchetti inutilmente aumenta la probabilità di errori.
        - Da capire come la distanza dal depot influenza quest azione

    I '+' stanno a simboleggiare un alto reward, i '-' invece un basso reward. Ovviamente la disposizione dipende dal 
    tipo di azione come descritto precedentemente.

    Un esempio per l'azione 2:
     ____ ____ ____ ____
    | -- |  - |  - | -- |
    |____|____|____|____|
     ____ ____ ____ ____
    |  - | +- | +- |  - |
    |____|____|____|____|
     ____ ____ ____ ____
    | +- |  + |  + | +- |
    |____|____|____|____|
     ____ ____ ____ ____
    |  + | ++ | ++ |  + |
    |____|____|____|____|

    """

    def __init__(self, drone, simulator):
        BASE_routing.__init__(self, drone, simulator)
        # random generator
        self.rnd_for_routing_ai = np.random.RandomState(self.simulator.seed)
        self.taken_actions = {}  # id event : (old_action)

        self.cell_number = pow(int(self.simulator.env_width / self.simulator.prob_size_cell), 2)
        self.action_number = 3  # we consider only the first 2 actions
        # self.q_value = [][]  # [N-cells][N-action] : N-actions =  0:send_pkt, 1:keep_pkt, 2:move_to_depot
        # self.q_value = np.array([[0 for i in range(self.action_number)] for j in range(self.cell_number)])
        self.q_value = [[0 for i in range(self.action_number)] for j in range(self.cell_number)]
        self.epsilon = 0.002
        self.alpha = 0.7
        self.gamma = 0.6
        self.to_depot = False

    def feedback(self, drone, id_event, delay, outcome):
        """ return a possible feedback, if the destination drone has received the packet """
        if config.DEBUG:
            # Packets that we delivered and still need a feedback
            print("Drone: ", self.drone.identifier, "---------- has delivered: ", self.taken_actions)

            # outcome == -1 if the packet/event expired; 0 if the packets has been delivered to the depot
            # Feedback from a delivered or expired packet
            print("Drone: ", self.drone.identifier, "---------- just received a feedback:",
                  "Drone:", drone, " - id-event:", id_event, " - delay:", delay, " - outcome:", outcome)

        # Be aware, due to network errors we can give the same event to multiple drones and receive multiple feedback
        # for the same packet!!
        # NOTE: reward or update using the old action!!
        # STORE WHICH ACTION DID YOU TAKE IN THE PAST.
        # do something or train the model (?)
        if id_event in self.taken_actions:
            action, old_cell, next_target_cell, mul_reward, time_to_depot= self.taken_actions[id_event]
            # if self.drone.identifier == 0:
            #     print(self.drone, " evento: ", id_event, " self.teken_actions:", self.taken_actions[id_event])
            # del self.taken_actions[id_event]

            # -- Test -- #
            # reward = (max_time - time_to_mission) / 100 * mul_reward
            
            # reward = -time_to_mission/100
            if action == 2:
                reward = (time_to_depot)/10 * mul_reward 
            else:
                # reward = ((self.simulator.event_duration - delay) / 1000) * mul_reward
                reward = mul_reward
            # print("REWARD", id_event, " action:", action, " reward:", reward, " cella:", old_cell, "next_target:", next_target_cell)

            # ---------- #
            # reward = ((self.simulator.event_duration - delay) / 1000) * mul_reward
            # reward = mul_reward * (outcome + 2)
            # reward = mul_reward
            self.to_depot = False
            self.q_value[old_cell][action] = self.q_value[old_cell][action] + self.alpha * (reward + self.gamma * max(self.q_value[next_target_cell]) - self.q_value[old_cell][action])

    def relay_selection(self, opt_neighbors, pkd):
        """ arg min score  -> geographical approach, take the drone closest to the depot """
        # Notice all the drones have different speed, and radio performance!!
        # you know the speed, not the radio performance.
        # self.drone.speed
        # Only if you need --> several features:
        cell_index = int(util.TraversedCells.coord_to_cell(size_cell=self.simulator.prob_size_cell,
                                                           width_area=self.simulator.env_width,
                                                           x_pos=self.drone.coords[0], y_pos=self.drone.coords[1])[0])


        action = None

        # now epsilon greedy selection of the action
        # 1) case epsilon, we take a random action
        if random.uniform(0, 1) < self.epsilon:
            action = random.randint(0, 2)
        # 2) case of 1 - epsilon we take the greatest q-value
        else:
            action = np.argmax(self.q_value[cell_index])

        # With self.drone.next_target() we know the next cell, i.e. the next state
        next_target_coord = self.drone.next_target()
        next_target_cell = None
        drone_to_send = None

        # -1 --> move to depot
        if action == 2:
            AIRouting.move_to_depot += 1
            drone_to_send = -1
            # Cell of depot
            next_target_cell = int(util.TraversedCells.coord_to_cell(size_cell=self.simulator.prob_size_cell,
                                                                     width_area=self.simulator.env_width,
                                                                     x_pos=self.simulator.depot_coordinates[0],
                                                                     y_pos=self.simulator.depot_coordinates[1])[0])

        # None --> no transmission, keep the packet
        if action == 1:
            AIRouting.keep_pkt += 1
            drone_to_send = None
            next_target_cell = int(util.TraversedCells.coord_to_cell(size_cell=self.simulator.prob_size_cell,
                                                                     width_area=self.simulator.env_width,
                                                                     x_pos=next_target_coord[0],
                                                                     y_pos=next_target_coord[1])[0])

        # send packet
        elif action == 0:
            AIRouting.send_pkt += 1
            drone_to_send = GeoRouting.relay_selection(self, opt_neighbors, pkd)
            if drone_to_send is None:
                action = 1
                AIRouting.send_pkt -= 1
                AIRouting.keep_pkt += 1
            next_target_cell = int(util.TraversedCells.coord_to_cell(size_cell=self.simulator.prob_size_cell,
                                                                     width_area=self.simulator.env_width,
                                                                     x_pos=next_target_coord[0],
                                                                     y_pos=next_target_coord[1])[0])

        mul_reward = self.calculate_reward(opt_neighbors, drone_to_send, cell_index, action)

        self.q_value[cell_index][action] = self.q_value[cell_index][action] + self.alpha * (
                mul_reward + self.gamma * max(self.q_value[next_target_cell]) - self.q_value[cell_index][action])

        # Store your current action --- you can add several stuff if needed to take a reward later
        time_to_depot = 0
        
        distance_from_depot = util.euclidean_distance(self.simulator.depot_coordinates, self.drone.coords)
        time_to_depot = distance_from_depot / self.drone.speed
        
        # Questo perchè ho visto che lo sovrascriveva e nel feedback non veniva mai stampata l'azione 2
        if action == 2:
            self.taken_actions[pkd.event_ref.identifier] = action, cell_index, next_target_cell, mul_reward, time_to_depot
            self.to_depot = True

        if self.to_depot == False:
            self.taken_actions[pkd.event_ref.identifier] = action, cell_index, next_target_cell, mul_reward, time_to_depot
        # print(self.drone, self.q_value)

        # return action:
        # None --> no transmission
        # -1 --> move to depot
        # 0, ... , self.ndrones --> send packet to this drone
        return drone_to_send  # here you should return a drone object!

    def calculate_reward(self, opt_neighbors, drone_to_send, cell_index, action):
        # We search if there is a drone that goes to the depot
        drones_to_depot = []
        for hpk, drone_instance in opt_neighbors:
            if drone_instance.next_target() == self.simulator.depot_coordinates:
                drones_to_depot.append(drone_instance)

        # We calculate the multiplier for the reward
        mul_reward = 0
        # Numero di celle su una riga    
        num_cell_in_row = int(self.simulator.env_width / self.simulator.prob_size_cell)
        depot_cell = int(util.TraversedCells.coord_to_cell(size_cell=self.simulator.prob_size_cell,
                                                           width_area=self.simulator.env_width,
                                                           x_pos=self.simulator.depot_coordinates[0], y_pos=self.simulator.depot_coordinates[1])[0])

        for row_number in range(num_cell_in_row):
            # in che riga mi trovo
            if cell_index < (row_number+1)*num_cell_in_row:
                if action == 0:
                    if drone_to_send in drones_to_depot:
                        mul_reward = 5
                    else:
                        mul_reward = (row_number+1) * abs(depot_cell*(row_number+1) - cell_index)
                        mul_reward = mul_reward * self.simulator.n_drones
                        break
                elif action == 1:
                    # Se avevo un vicino che gia andava al depot darò un reward molto basso
                    if len(drones_to_depot) != 0:
                        mul_reward = -5
                    else:
                        # moltiplico la riga per il valore assoluto della differenza tra la cella del depot e quella del drone
                        # Più mi allontano più basso sarà il reward
                        mul_reward = (num_cell_in_row - row_number) * abs(depot_cell*(row_number+1) - cell_index)
                        break
                elif action == 2:
                    # Se avevo un vicino che gia andava al depot darò un reward molto basso
                    if len(drones_to_depot) != 0:
                        mul_reward = -5
                    else:
                        # moltiplico la riga per il valore assoluto della differenza tra la cella del depot e quella del drone
                        # Più mi allontano più basso sarà il reward
                        mul_reward = - (num_cell_in_row - row_number) * abs(depot_cell*(row_number+1) - cell_index)
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
        print("Send the Packet: ", AIRouting.send_pkt)
        print("Keep the Packet: ", AIRouting.keep_pkt)
        print("Move to Depot: ", AIRouting.move_to_depot)
        print("####################################\n")
import numpy as np

from src.routing_algorithms.georouting_w_move import GeoMoveRouting
from src.routing_algorithms.georouting import GeoRouting
from src.utilities import config
from src.utilities import utilities as util
from src.routing_algorithms.BASE_routing import BASE_routing
import random


class AIRouting(BASE_routing):
    keep_pkt = 0
    send_pkt = 0
    move_to_depot = 0

    """
    Il buon funzionamento dell'algoritmo credo dipenda molto dalla posizione in cui effettuo l'azione e che tipo di 
    azione. Per questo credo che le celle siano un ottimo modo per strutturare l'algoritmo.

    Ogni cella ha un array di 3 valori corrispondente alle azioni.
    - Azione 2:
        - Più sono lontano dal depot più l'azione 2  deve essere negativa.
        - Più sono vicino al depot più l'azione 2  deve essere positiva.

    - Azione 1 (keep_pkt):
        - Più sono lontano dal depot più l'azione 1 deve essere positiva.
        - Più sono vicino al depot più l'azione 2 deve essere negativa.
            Questo perchè se sono vicino conviene tentare di consegnare il pacchetto
        - Se tra i vicini ho un drone diretto al depot e scelgo l'azione 1 devo avere un reward molto negativo.

    - Azione 0 (send_pkt):
        - Invio esclusivamente a droni diretti verso il depot, altrimenti faccio l'azione 1, questo perchè
            inviare pacchetti inutilmente aumenta la probabilità di errori.
        - Da capire come la distanza dal depot influenza quest azione

    I '+' stanno a simboleggiare un alto reward, i '-' invece un basso reward. Ovviamente la disposizione dipende dal 
    tipo di azione come descritto precedentemente.

    Un esempio per l'azione 2:
     ____ ____ ____ ____
    | -- |  - |  - | -- |
    |____|____|____|____|
     ____ ____ ____ ____
    |  - | +- | +- |  - |
    |____|____|____|____|
     ____ ____ ____ ____
    | +- |  + |  + | +- |
    |____|____|____|____|
     ____ ____ ____ ____
    |  + | ++ | ++ |  + |
    |____|____|____|____|

    """

    def __init__(self, drone, simulator):
        BASE_routing.__init__(self, drone, simulator)
        # random generator
        self.rnd_for_routing_ai = np.random.RandomState(self.simulator.seed)
        self.taken_actions = {}  # id event : (old_action)

        self.cell_number = pow(int(self.simulator.env_width / self.simulator.prob_size_cell), 2)
        self.action_number = 3  # we consider only the first 2 actions
        # self.q_value = [][]  # [N-cells][N-action] : N-actions =  0:send_pkt, 1:keep_pkt, 2:move_to_depot
        # self.q_value = np.array([[0 for i in range(self.action_number)] for j in range(self.cell_number)])
        self.q_value = [[0 for i in range(self.action_number)] for j in range(self.cell_number)]
        self.epsilon = 0.002
        self.alpha = 0.7
        self.gamma = 0.6
        self.to_depot = False

    def feedback(self, drone, id_event, delay, outcome):
        """ return a possible feedback, if the destination drone has received the packet """
        if config.DEBUG:
            # Packets that we delivered and still need a feedback
            print("Drone: ", self.drone.identifier, "---------- has delivered: ", self.taken_actions)

            # outcome == -1 if the packet/event expired; 0 if the packets has been delivered to the depot
            # Feedback from a delivered or expired packet
            print("Drone: ", self.drone.identifier, "---------- just received a feedback:",
                  "Drone:", drone, " - id-event:", id_event, " - delay:", delay, " - outcome:", outcome)

        # Be aware, due to network errors we can give the same event to multiple drones and receive multiple feedback
        # for the same packet!!
        # NOTE: reward or update using the old action!!
        # STORE WHICH ACTION DID YOU TAKE IN THE PAST.
        # do something or train the model (?)
        if id_event in self.taken_actions:
            action, old_cell, next_target_cell, mul_reward, time_to_depot= self.taken_actions[id_event]
            # if self.drone.identifier == 0:
            #     print(self.drone, " evento: ", id_event, " self.teken_actions:", self.taken_actions[id_event])
            # del self.taken_actions[id_event]

            # -- Test -- #
            # reward = (max_time - time_to_mission) / 100 * mul_reward
            
            # reward = -time_to_mission/100
            if action == 2:
                reward = (time_to_depot)/10 * mul_reward 
            else:
                # reward = ((self.simulator.event_duration - delay) / 1000) * mul_reward
                reward = mul_reward
            # print("REWARD", id_event, " action:", action, " reward:", reward, " cella:", old_cell, "next_target:", next_target_cell)

            # ---------- #
            # reward = ((self.simulator.event_duration - delay) / 1000) * mul_reward
            # reward = mul_reward * (outcome + 2)
            # reward = mul_reward
            self.to_depot = False
            self.q_value[old_cell][action] = self.q_value[old_cell][action] + self.alpha * (reward + self.gamma * max(self.q_value[next_target_cell]) - self.q_value[old_cell][action])

    def relay_selection(self, opt_neighbors, pkd):
        """ arg min score  -> geographical approach, take the drone closest to the depot """
        # Notice all the drones have different speed, and radio performance!!
        # you know the speed, not the radio performance.
        # self.drone.speed
        # Only if you need --> several features:
        cell_index = int(util.TraversedCells.coord_to_cell(size_cell=self.simulator.prob_size_cell,
                                                           width_area=self.simulator.env_width,
                                                           x_pos=self.drone.coords[0], y_pos=self.drone.coords[1])[0])


        action = None

        # now epsilon greedy selection of the action
        # 1) case epsilon, we take a random action
        if random.uniform(0, 1) < self.epsilon:
            action = random.randint(0, 2)
        # 2) case of 1 - epsilon we take the greatest q-value
        else:
            action = np.argmax(self.q_value[cell_index])

        # With self.drone.next_target() we know the next cell, i.e. the next state
        next_target_coord = self.drone.next_target()
        next_target_cell = None
        drone_to_send = None

        # -1 --> move to depot
        if action == 2:
            AIRouting.move_to_depot += 1
            drone_to_send = -1
            # Cell of depot
            next_target_cell = int(util.TraversedCells.coord_to_cell(size_cell=self.simulator.prob_size_cell,
                                                                     width_area=self.simulator.env_width,
                                                                     x_pos=self.simulator.depot_coordinates[0],
                                                                     y_pos=self.simulator.depot_coordinates[1])[0])

        # None --> no transmission, keep the packet
        if action == 1:
            AIRouting.keep_pkt += 1
            drone_to_send = None
            next_target_cell = int(util.TraversedCells.coord_to_cell(size_cell=self.simulator.prob_size_cell,
                                                                     width_area=self.simulator.env_width,
                                                                     x_pos=next_target_coord[0],
                                                                     y_pos=next_target_coord[1])[0])

        # send packet
        elif action == 0:
            AIRouting.send_pkt += 1
            drone_to_send = GeoRouting.relay_selection(self, opt_neighbors, pkd)
            if drone_to_send is None:
                action = 1
                AIRouting.send_pkt -= 1
                AIRouting.keep_pkt += 1
            next_target_cell = int(util.TraversedCells.coord_to_cell(size_cell=self.simulator.prob_size_cell,
                                                                     width_area=self.simulator.env_width,
                                                                     x_pos=next_target_coord[0],
                                                                     y_pos=next_target_coord[1])[0])

        mul_reward = self.calculate_reward(opt_neighbors, drone_to_send, cell_index, action)

        self.q_value[cell_index][action] = self.q_value[cell_index][action] + self.alpha * (
                mul_reward + self.gamma * max(self.q_value[next_target_cell]) - self.q_value[cell_index][action])

        # Store your current action --- you can add several stuff if needed to take a reward later
        time_to_depot = 0
        
        distance_from_depot = util.euclidean_distance(self.simulator.depot_coordinates, self.drone.coords)
        time_to_depot = distance_from_depot / self.drone.speed
        
        # Questo perchè ho visto che lo sovrascriveva e nel feedback non veniva mai stampata l'azione 2
        if action == 2:
            self.taken_actions[pkd.event_ref.identifier] = action, cell_index, next_target_cell, mul_reward, time_to_depot
            self.to_depot = True

        if self.to_depot == False:
            self.taken_actions[pkd.event_ref.identifier] = action, cell_index, next_target_cell, mul_reward, time_to_depot
        # print(self.drone, self.q_value)

        # return action:
        # None --> no transmission
        # -1 --> move to depot
        # 0, ... , self.ndrones --> send packet to this drone
        return drone_to_send  # here you should return a drone object!

    def calculate_reward(self, opt_neighbors, drone_to_send, cell_index, action):
        # We search if there is a drone that goes to the depot
        drones_to_depot = []
        for hpk, drone_instance in opt_neighbors:
            if drone_instance.next_target() == self.simulator.depot_coordinates:
                drones_to_depot.append(drone_instance)

        # We calculate the multiplier for the reward
        mul_reward = 0
        # Numero di celle su una riga    
        num_cell_in_row = int(self.simulator.env_width / self.simulator.prob_size_cell)
        depot_cell = int(util.TraversedCells.coord_to_cell(size_cell=self.simulator.prob_size_cell,
                                                           width_area=self.simulator.env_width,
                                                           x_pos=self.simulator.depot_coordinates[0], y_pos=self.simulator.depot_coordinates[1])[0])

        for row_number in range(num_cell_in_row):
            # in che riga mi trovo
            if cell_index < (row_number+1)*num_cell_in_row:
                if action == 0:
                    if drone_to_send in drones_to_depot:
                        mul_reward = 5
                    else:
                        mul_reward = (row_number+1) * abs(depot_cell*(row_number+1) - cell_index)
                        mul_reward = mul_reward * self.simulator.n_drones
                        break
                elif action == 1:
                    # Se avevo un vicino che gia andava al depot darò un reward molto basso
                    if len(drones_to_depot) != 0:
                        mul_reward = -5
                    else:
                        # moltiplico la riga per il valore assoluto della differenza tra la cella del depot e quella del drone
                        # Più mi allontano più basso sarà il reward
                        mul_reward = (num_cell_in_row - row_number) * abs(depot_cell*(row_number+1) - cell_index)
                        break
                elif action == 2:
                    # Se avevo un vicino che gia andava al depot darò un reward molto basso
                    if len(drones_to_depot) != 0:
                        mul_reward = -5
                    else:
                        # moltiplico la riga per il valore assoluto della differenza tra la cella del depot e quella del drone
                        # Più mi allontano più basso sarà il reward
                        mul_reward = - (num_cell_in_row - row_number) * abs(depot_cell*(row_number+1) - cell_index)
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
        print("Send the Packet: ", AIRouting.send_pkt)
        print("Keep the Packet: ", AIRouting.keep_pkt)
        print("Move to Depot: ", AIRouting.move_to_depot)
        print("####################################\n")
