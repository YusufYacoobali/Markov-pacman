# mdpAgents.py
# parsons/20-nov-2017
#
# Version 1
#
# The starting point for CW2.
#
# Intended to work with the PacMan AI projects from:
#
# http://ai.berkeley.edu/
#
# These use a simple API that allow us to control Pacman's interaction with
# the environment adding a layer on top of the AI Berkeley code.
#
# As required by the licensing agreement for the PacMan AI we have:
#
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

# The agent here is was written by Simon Parsons, based on the code in
# pacmanAgents.py

from pacman import Directions
from game import Agent
import api
# import random
# import game
# import util

# RUN THIS COMMAND:  python pacman.py -q -n 25 -p MDPAgent -l mediumClassic

#constants
CAPSULE_REWARD = 4
GHOST_EDIBLE_REWARD = 20
WALL_REWARD = 0
EMPTY_SQUARE_REWARD = 0

class MDPAgent(Agent):

    # Constructor: this gets run when we first invoke pacman.py
    def __init__(self):
        print "Starting up MDPAgent!"
        name = "Pacman"
        self.map = None
        self.width = 0
        self.height = 0
        self.prev_map = None
        self.walls = None
        self.corners = None
        self.small_grid = False
        self.default_direction = Directions.EAST

    # Gets run after an MDPAgent object is created and once there is
    # game state to access.
    def registerInitialState(self, state):
        # print "Running registerInitialState for MDPAgent!"
        # print "I'm at:"
        # print api.whereAmI(state)
        self.walls, self.corners = api.walls(state), api.corners(state)
        self.width, self.height = self.corners[3][0] + 1, self.corners[3][1] + 1
        #Small grid has different values for constants
        if self.width < 8:
            self.small_grid = True
            self.FOOD_REWARD = 1
            self.GHOST_REWARD = -5
            self.ITERATIONS = 8
            self.DISCOUNT = 0.7
            self.RADIUS = 2
        else:
            self.FOOD_REWARD = 2
            self.GHOST_REWARD = -10
            self.ITERATIONS = 25
            self.DISCOUNT = 0.4
            self.RADIUS = 3
        self.createMap(state)
        self.prev_map = self.map

    # This is what gets run in between multiple games
    def final(self, state):
        print "Looks like the game just ended!"

    # Picks the best action after value iteration is applied
    def getAction(self, state):
        legal = api.legalActions(state)
        pacman = api.whereAmI(state)

        if Directions.STOP in legal:
            legal.remove(Directions.STOP)
        # Make the map if its not done 
        if self.map == None:
            self.registerInitialState(state)
        self.valueIteration(state)
        best_move = self.getBestMove(pacman[0], pacman[1], legal)
        return api.makeMove(best_move, legal)
    
    # Perform value iteration as many times as the constant
    def valueIteration(self, state):
        # Put fresh values into the current map
        self.updateMapValues(state)
        for i in range(self.ITERATIONS):
            # Make a map copy so that you dont use new values in a round of value iteration
            self.prev_map = [row[:] for row in self.map]

            for y in range(self.height):
                for x in range(self.width):
                    if (x, y) not in self.walls: 
                        new_value = self.bellmanValue(state, x, y, self.prev_map)
                        self.map[y][x] = new_value

    # Picks the max value for each cell unless a ghost is nearby
    def bellmanValue(self, state, x, y, prevMap):
        all_directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        cell_values = []

        for x2, y2 in all_directions:
            main_x, main_y = self.handleWallBounce(x, y, x + x2, y + y2)
            right_x, right_y = self.handleWallBounce(x, y, x + y2, y + x2)
            left_x, left_y = self.handleWallBounce(x, y, x - y2, y - x2)

            values = [
                0.8 * prevMap[main_y][main_x],
                0.1 * prevMap[right_y][right_x],
                0.1 * prevMap[left_y][left_x]
            ]
            cell_values.append(sum(values))

        negative_values = [value for value in cell_values if value < 0]
        # if its a small grid simply return the bellman value, if there is a negative value from the ghosts then return that
        if self.small_grid:
            return prevMap[y][x] + self.DISCOUNT * min(negative_values) if (negative_values) else prevMap[y][x] + self.DISCOUNT * max(cell_values)
        else:
            # if the cell is within a ghosts radius then pick the worst possible value so pacman doesnt go there
            if self.isWithinRadius(state, x, y):
                val = min(cell_values)
            else:
                val = max(cell_values)
            return prevMap[y][x] + self.DISCOUNT * val
    
    # if pacman would hit a wall with the intended direction then return the original cell
    def handleWallBounce(self, x, y, main_x, main_y):
        return (x, y) if (main_x, main_y) in self.walls else (main_x, main_y)
    
    # if the coordinate is near a ghost then return true
    def isWithinRadius(self, state, x, y):
        ghost_positions = api.ghosts(state)
        ghostsState = api.ghostStates(state)
        for (ghost_x, ghost_y), ghost_state in zip(ghost_positions, ghostsState):
            if ghost_state[1] != 1:  # Check if the ghost is not edible
                ghost_distance = abs(x - ghost_x) + abs(y - ghost_y)
                if ghost_distance <= self.RADIUS:
                    return True
        return False

    # Picks the best move from the value iteration grid
    def getBestMove(self, x, y, legal):
        legal_directions = [(0, 1) if Directions.NORTH in legal else (0, 0),
                            (0, -1) if Directions.SOUTH in legal else (0, 0),
                            (1, 0) if Directions.EAST in legal else (0, 0),
                            (-1, 0) if Directions.WEST in legal else (0, 0)]

        # Filter out directions so only legal moves are looked at
        legal_directions = [(x2, y2) for x2, y2 in legal_directions if (x2, y2) != (0, 0)]
        # Get max value from legal directions applied on the map
        best_move = max(legal_directions, key=lambda b: self.map[y + b[1]][x + b[0]])

        # If the best move is nothing at all then use the default direction logic
        if self.map[y + best_move[1]][x + best_move[0]] == 0:
            self.default_direction = self.default_direction if self.default_direction in legal else legal[-1]
            return self.default_direction

        if best_move == (0, 1):
            return Directions.NORTH
        elif best_move == (0, -1):
            return Directions.SOUTH
        elif best_move == (1, 0):
            return Directions.EAST
        else:  
            return Directions.WEST

    # Used initially to create the map
    def createMap(self,state):
        self.map = []
        for _ in range(self.height):
            row = [0] * self.width
            self.map.append(row)
        self.updateMapValues(state)

    # Update map with fresh values before value iteration occurs
    def updateMapValues(self, state):
        foods = set(api.food(state))
        ghosts = api.ghosts(state)
        ghostsStates = api.ghostStates(state)
        capsules = set(api.capsules(state))

        for y, row in enumerate(self.map):
            for x, value in enumerate(row):
                position = (x, y)

                if position in self.walls:
                    self.map[y][x] = WALL_REWARD
                elif position in ghosts:
                    ghost_index = ghosts.index(position)
                    ghost_state = ghostsStates[ghost_index][1] if ghost_index < len(ghostsStates) else 0
                    self.map[y][x] = GHOST_EDIBLE_REWARD if ghost_state == 1 else self.GHOST_REWARD
                elif position in foods:
                    self.map[y][x] = self.FOOD_REWARD
                elif position in capsules:
                    self.map[y][x] = CAPSULE_REWARD
                else:
                    self.map[y][x] = EMPTY_SQUARE_REWARD