
import numpy as np
from src.utilities import utilities as util
from src.routing_algorithms.BASE_routing import BASE_routing
from matplotlib import pyplot as plt

from random import randrange

class AIRouting(BASE_routing):
    def __init__(self, drone, simulator):
        BASE_routing.__init__(self, drone, simulator)
        # random generator
        self.rnd_for_routing_ai = np.random.RandomState(self.simulator.seed)
        self.taken_actions = {}  #id event : (old_action)

    def feedback(self, drone, id_event, delay, outcome):
        """ return a possible feedback, if the destination drone has received the packet """
        # Packets that we delivered and still need a feedback
        print(self.drone.identifier, "----------", self.taken_actions)

        # outcome == -1 if the packet/event expired; 0 if the packets has been delivered to the depot
        # Feedback from a delivered or expired packet
        print(self.drone.identifier, "----------", drone, id_event, delay, outcome)

        # Be aware, due to network errors we can give the same event to multiple drones and receive multiple feedback for the same packet!!
        # NOTE: reward or update using the old action!!
        # STORE WHICH ACTION DID YOU TAKE IN THE PAST.
        # do something or train the model (?)
        if id_event in self.taken_actions:
            action = self.taken_actions[id_event]
            del self.taken_actions[id_event]

    def relay_selection(self, opt_neighbors, pkd):
        """ arg min score  -> geographical approach, take the drone closest to the depot """

        # Only if you need --> several features:
        cell_index = util.TraversedCells.coord_to_cell(size_cell=self.simulator.prob_size_cell,
                                                        width_area=self.simulator.env_width,
                                                        x_pos=self.drone.coords[0],  # e.g. 1500
                                                        y_pos=self.drone.coords[1])[0]  # e.g. 500
        # print("Cell index", cell_index)
        
        # ----- TEST ----- #
        action = None
        drone_to_send = None
        reward = None 
        # I Droni "traghetto" sono 0, 1, 2, invece i Droni che possono accumulare 
        # informazioni e decidere se trasmettere i pacchetti sono 3, 4.

        # Per prima cosa dobbiamo accertarci che la decisione se mandare i 
        # pacchetti o meno sia effettuata da uno dei droni 3 o 4.
        if not self.drone.identifier in [0, 1, 2]:
            # Creiamo una lista di tutti i droni nelle vicinanze
            neighbors_drone = []
            for hll_pck, drone in opt_neighbors:
                neighbors_drone.append(drone)

            # Controlliamo se l'evento è già presente nel dizionario (cronologia)
            if pkd.event_ref.identifier in self.taken_actions:
                # DOMANDA: Se l'azione precedente è andata a buon fine, come e perchè 
                # devo cambiare drone ???
                # RISPOSTA: (Se avete capito, io no... ) 
                # print("IDENTIFIER: ", pkd.event_ref.identifier)

                # TODO: Implementare reward. Per ora rifaccio la stessa azione.
                drone_to_send = self.taken_actions[pkd.event_ref.identifier][1]
            else:
                # Un primo approccio (corretto??) è quello di scelgiere randomicamente 
                # il drone a cui affidare i dati, in quanto gli Action Value (Q) sono
                # tutti a 0 di default.
                # Altri possibili approcci possibili: GeoR, ...?
                reward = 0
                drone_to_send = neighbors_drone[randrange(len(neighbors_drone))]

            # Considero 'action' come una tupla contenente il reward e il drone scelto    
            action = (reward, drone_to_send)
            # print("ACTION: ", action)
        

        # self.drone.history_path (which waypoint I traversed. We assume the mission is repeated)
        # self.drone.residual_energy (that tells us when I'll come back to the depot).
        #  .....

        # Store your current action --- you can add several stuff if needed to take a reward later
        self.taken_actions[pkd.event_ref.identifier] = (action)
        
        #print("DIZIONARIO ", self.taken_actions)
        
        return None
        #return drone_to_send  # here you should return a drone object!

    def print(self):
        """
            This method is called at the end of the simulation, can be usefull to print some
                metrics about the learning process
        """
        pass
