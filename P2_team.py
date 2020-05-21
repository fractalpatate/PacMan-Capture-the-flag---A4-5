# myTeam.py
# ---------
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


from captureAgents import CaptureAgent
import random, time, util
from game import Directions
import game

# My imports
import numpy as np
from game import Actions
from scipy.special import softmax
from math import inf

# Constants
MOVE_DIRECTIONS = set(['North', 'South', 'East', 'West'])

def manhattan_distance(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

def createTeam(firstIndex, secondIndex, isRed, first = 'OffensiveEMMAgent', second = 'DefensiveEMMAgent'):
  """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.
  """
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

class EMMAgent(CaptureAgent):

  def registerInitialState(self, gameState):
    """
    This method handles the initial setup of the
    agent to populate useful fields (such as what team
    we're on).

    A distanceCalculator instance caches the maze distances
    between each pair of positions, so your agents can use:
    self.distancer.getDistance(p1, p2)

    IMPORTANT: This method may run for at most 15 seconds.
    """

    '''
    Make sure you do not delete the following line. If you would like to
    use Manhattan distances instead of maze distances in order to save
    on initialization time, please take a look at
    CaptureAgent.registerInitialState in captureAgents.py.
    '''
    CaptureAgent.registerInitialState(self, gameState)

    # Opponents
    self.opponent_indices = self.getOpponents(gameState)
    self.opponent_index_to_pos = {index: i for i, index in enumerate(self.opponent_indices)}
    self.n_opponents = len(self.opponent_indices)

    # Markov model for opponent position 

    # Walls
    walls = gameState.getWalls()
    walls_list = walls.asList()

    # Grid
    width, height = walls.width, walls.height
    grid_list = [pos for pos in np.ndindex((width, height))]

    # Non-wall positions
    self.pos_list = [pos for pos in grid_list if pos not in walls_list]

    # Markov model states
    self.n = len(self.pos_list)
    # Position probability vectors
    self.opponent_pos_vecs = {opponent_index: np.zeros(self.n) for opponent_index in self.opponent_indices}

    # Maps state index to position (np.array NOT tuple!) 
    self.state_idx_to_pos = np.empty(self.n, dtype='O')
    self.state_idx_to_pos[:] = self.pos_list
    # Maps position tuple to state index
    self.pos_to_state_idx = {pos: i for i, pos in enumerate(self.pos_list)}

    # Set initial position probability distributions
    for opponent_index in self.opponent_indices:
      opponent_start_pos = gameState.getInitialAgentPosition(opponent_index)
      self.opponent_pos_vecs[opponent_index][self.pos_to_state_idx[opponent_start_pos]] = 1

    # Markov model transition matrix
    self.transition_matrix = np.zeros((self.n, self.n))
    for i in range(self.n):
      pos = self.state_idx_to_pos[i]
      # No STOP, agent must move
      neighbors = Actions.getLegalNeighbors(pos, walls)[:-1]
      # Equal probability amongst neighbors
      p = 1 / len(neighbors)
      for pos_neighbor in neighbors:
        j = self.pos_to_state_idx[pos_neighbor]
        self.transition_matrix[i][j] = p

  def get_possible_opponent_positions(self, opponent_index, gameState):
    # Get distance to opponent
    distances = gameState.getAgentDistances()
    opponent_distance = distances[opponent_index]
    own_pos = gameState.getAgentPosition(self.index)

    # Get food you're defending 
    own_food = self.getFoodYouAreDefending(gameState)

    return np.array([self.pos_to_state_idx[pos] for pos in self.pos_list 
      if gameState.getDistanceProb(manhattan_distance(pos, own_pos), opponent_distance) > 0
      and not own_food[pos[0]][pos[1]]])

  def update_opponent_pos(self, opponent_index, gameState):
    opponent_pos = gameState.getAgentPosition(opponent_index)

    # Check if position is known
    if opponent_pos is not None:
      self.opponent_pos_vecs[opponent_index] = np.zeros(self.n)
      self.opponent_pos_vecs[opponent_index][self.pos_to_state_idx[opponent_pos]] = 1
      return

    # Check if opponent pacman was eaten => reset to start position (need to check for None at first move)
    prev_state = self.getPreviousObservation()
    if prev_state is not None: 
      prev_opponent_pos = prev_state.getAgentPosition(opponent_index)
      if prev_opponent_pos is not None and self.get_position(gameState) == prev_opponent_pos:
        self.opponent_pos_vecs[opponent_index] = np.zeros(self.n)
        self.opponent_pos_vecs[opponent_index][self.pos_to_state_idx[gameState.getInitialAgentPosition(opponent_index)]] = 1
        return

    # Move Markov model forward
    opponent_pos_vec = self.opponent_pos_vecs[opponent_index] @ self.transition_matrix

    # Get all points which are within distance and convert to state indices
    possible_pos = self.get_possible_opponent_positions(opponent_index, gameState)

    # Update opponent position based on possible positions and Markov model
    updated_pos = np.zeros(self.n)
    updated_pos[possible_pos] = opponent_pos_vec[possible_pos]
    self.opponent_pos_vecs[opponent_index] = updated_pos

  def update_opponent_positions(self, gameState):
    for opponent_index in self.opponent_indices:
      self.update_opponent_pos(opponent_index, gameState)

  def get_opponent_positions(self):
    return {opponent_index: self.state_idx_to_pos[np.argmax(self.opponent_pos_vecs[opponent_index])] for opponent_index in self.opponent_indices}

  def get_opponent_position(self, opponent_index):
    return self.get_opponent_positions()[opponent_index]

  def get_state(self, gameState):
    return gameState.getAgentState(self.index)

  def get_position(self, gameState):
    return gameState.getAgentPosition(self.index)

  def get_distance_to_closest_opponent(self, gameState):
    agent_pos = self.get_position(gameState)
    opponent_positions = self.get_opponent_positions()
    return min([self.getMazeDistance(agent_pos, opponent_positions[opponent_index]) for opponent_index in self.opponent_indices])

  def utility(self, gameState):
    prev_state = self.getPreviousObservation()
    if prev_state is not None and self.get_position(gameState) == self.get_position(prev_state):
      # Going back to previous position is pointless
      return -inf
    return np.dot(self.get_feature_vec(gameState), self.get_weight_vec(gameState))

  def expectiminimax(self, node, depth, gameState):
    return max((self.utility(gameState.generateSuccessor(self.index, action)), action) for action in gameState.getLegalActions(self.index) if action in MOVE_DIRECTIONS)

  def chooseAction(self, gameState):
    self.update_opponent_positions(gameState)
    utility, action = self.expectiminimax('self', 4, gameState)
    return action

class OffensiveEMMAgent(EMMAgent):

  def get_distance_to_closest_food(self, gameState):
    pos = self.get_position(gameState)
    food = self.getFood(gameState).asList()
    if not food:
      return inf
    return min(self.getMazeDistance(pos, food_pos) for food_pos in food)

  def get_distance_to_closest_opponent_ghost(self, gameState):
    agent_pos = self.get_position(gameState)
    ghost_opponent_indices = pacman_opponent_indices = [opponent_index for opponent_index in self.opponent_indices if not gameState.getAgentState(opponent_index).isPacman]
    if not ghost_opponent_indices:
      return 0
    opponent_positions = self.get_opponent_positions()
    return min(self.getMazeDistance(agent_pos, opponent_positions[opponent_index]) for opponent_index in ghost_opponent_indices)

  def get_feature_vec(self, gameState):
    agent_pos = self.get_position(gameState)
    agent_pos = (int(agent_pos[0]), int(agent_pos[1]))

    # Number of food you're eating
    food_count = len(self.getFood(gameState).asList())

    # Distance to closest food
    closest_food_distance = self.get_distance_to_closest_food(gameState)

    # Distance to return food
    distance_to_return = self.getMazeDistance(agent_pos, gameState.getInitialAgentPosition(self.index))

    # Distance to closest opponent ghost
    closest_opponent_ghost_distance = self.get_distance_to_closest_opponent_ghost(gameState)

    # Capsules you can eat
    capsules_count = len(self.getCapsules(gameState))

    return [closest_food_distance, food_count, distance_to_return, closest_opponent_ghost_distance, capsules_count]

  def get_weight_vec(self, gameState):
    closest_food_factor = -1
    food_count_factor = -30
    distance_to_return_factor = -20 if self.get_state(gameState).numCarrying > 5 else 0
    closest_opponent_ghost_distance_factor = 1 if self.get_state(gameState).isPacman else 0
    capsule_factor = 10
    return [closest_food_factor, food_count_factor, distance_to_return_factor, closest_opponent_ghost_distance_factor, capsule_factor]

class DefensiveEMMAgent(EMMAgent):

  def get_distance_to_closest_opponent_pacman(self, gameState):
    agent_pos = self.get_position(gameState)
    pacman_opponent_indices = [opponent_index for opponent_index in self.opponent_indices if gameState.getAgentState(opponent_index).isPacman]
    if not pacman_opponent_indices:
      return 0
    opponent_positions = self.get_opponent_positions()
    return min(self.getMazeDistance(agent_pos, opponent_positions[opponent_index]) for opponent_index in pacman_opponent_indices)

  def get_feature_vec(self, gameState):
    agent_pos = self.get_position(gameState)
    agent_pos = (int(agent_pos[0]), int(agent_pos[1]))

    # Pacman => on opponent side 
    on_opponent_side = gameState.getAgentState(self.index).isPacman

    # Number of enemy pacman
    opponent_pacman_count = sum(gameState.getAgentState(opponent_index).isPacman for opponent_index in self.opponent_indices)

    # Distance to closest opponent pacman
    closest_opponent_pacman_distance = self.get_distance_to_closest_opponent_pacman(gameState)

    # Distance to closest opponent
    closest_opponent_distance = self.get_distance_to_closest_opponent(gameState)

    return [on_opponent_side, opponent_pacman_count, closest_opponent_pacman_distance, closest_opponent_distance]

  def get_weight_vec(self, gameState):
    on_opponent_side_factor = -10
    opponent_pacman_count_factor = -50
    closest_opponent_pacman_distance = -5
    closest_opponent_distance_factor = 10 if self.get_state(gameState).scaredTimer > 0 else -1
    return [on_opponent_side_factor, opponent_pacman_count_factor, closest_opponent_pacman_distance, closest_opponent_distance_factor]

