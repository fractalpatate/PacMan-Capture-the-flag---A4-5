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


#NFSP imports
from keras.models import Sequential
from keras.models import Model, Sequential
from keras.layers import Dense, Input, Conv2D, MaxPool2D, LeakyReLU, ReLU, Flatten, Dropout
from keras.optimizers import Adam, SGD
from keras.losses import categorical_crossentropy, Huber
from keras.callbacks import ModelCheckpoint
# from NFSP.memory import ReplayMemory, ReservoirMemory
import numpy as np
from keras.callbacks import TensorBoard
import keras.backend as K
import tensorflow as tf
from keras.layers.advanced_activations import LeakyReLU
import configparser
import os

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'DummyAgent', second = 'DummyAgent'):
  """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.

  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, hiowever, your team will be created wthout
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """

  # The following line is an example only; feel free to change it.
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########


class NFSPModel:
  #ADAPTED FROM https://github.com/contr4l/Neural-Ficititious-Self-Play-in-Imperfect-Information-Games/
  target_br_model, best_response_model, average_response_model = None, None, None
  rl_memory, sl_memory = None, None
  sgd_br = SGD(lr=0.1)
  sgd_ar = SGD(lr=0.1)
  init = False
  AVG_POLICY, BR_POLICY = 'AVG', 'BR'
  game_step = 0
  EPSILON = 0
  ITERATION = 0
  def __init__(self, idx, state_shape, action_shape):
    self.idx = idx
    self.s_dim = state_shape
    self.a_dim = action_shape

    NFSPModel.average_response_model = self.__build_average_response_model()
    NFSPModel.best_response_model = self.__build_best_response_model()
    NFSPModel.target_br_model = self.__build_best_response_model()

  def __build_average_response_model(self):
    model = Sequential()
    model.add(Conv2D(filters=32, activation='relu', kernel_size=(5,5), padding="Same", input_shape=self.s_dim))
    model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='Same', activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='Same'))
    model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding = 'Same'))
    model.add(MaxPool2D(pool_size=(2, 2), strides= (2,2)))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(self.a_dim[0], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=NFSPModel.sgd_ar, metrics=['accuracy', 'crossentropy'])
    model.load_weights("models/ar_1.ckpt")
    return model

  def __build_best_response_model(self):
    model = Sequential()
    model.add(Conv2D(filters=32, activation='relu', kernel_size=(5,5), padding="Same", input_shape=self.s_dim))
    model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='Same', activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='Same'))
    model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding = 'Same'))
    model.add(MaxPool2D(pool_size=(2, 2), strides= (2,2)))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(self.a_dim[0], activation='softmax'))
    model.compile(optimizer=NFSPModel.sgd_br, loss=tf.keras.losses.Huber(), metrics=['accuracy', 'mse'])
    model.load_weights("models/br_1.ckpt")
    return model


  def act_best_response(self, state):
    return NFSPModel.best_response_model.predict(state)

  def sampleAction(self, policy, state, terminal):
    NFSPModel.game_step += 1
    ret = (None, terminal)
    if not terminal:
      if policy == NFSPModel.AVG_POLICY:
        a = NFSPModel.average_response_model.predict(state)
      else:
        a = self.act_best_response(state)
      ret = (a, terminal)
    return ret


class DummyAgent(CaptureAgent):
  """
  A Dummy agent to serve as an example of the necessary agent structure.
  You should look at baselineTeam.py for more details about how to
  create an agent as this is the bare minimum.
  """

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

    self.model = NFSPModel(self.index, (16, 32, 3), (4,))
    self.st = None #last state
    self.at = None #last action
    self.terminal = False
    self.reward = 0
    self.actions = {0 : 'North', 1 : 'South', 2 : 'East', 3 : 'West', 4 : 'Stop'}
    self.trans_actions_map = {'North' : 'South', 'South' : 'North', 'West' : 'East', 'East' : 'West', 'Stop' : 'Stop'}
    self.trans_actions = {0 : 1, 1 : 0, 2 : 3, 3 : 2, 4 : 4}
    self.policy = np.random.choice([NFSPModel.AVG_POLICY, NFSPModel.BR_POLICY], p = [0, 1])
    self.cnt = 0
    self.numRedFood = len(gameState.getRedFood().asList())
    self.numBlueFood = len(gameState.getBlueFood().asList())

  def _trans(self, inp):
    a, b = np.flip(inp[:,inp.shape[1]//2 :], 1).copy(), np.flip(inp[:,: inp.shape[1]//2], 1).copy()
    inp[:,: inp.shape[1]//2] = a
    inp[:,inp.shape[1]//2 :] = b

  def _convert_state_to_input(self, gameState, index):
    width, height = gameState.data.layout.width, gameState.data.layout.height
    inp1 = np.zeros((height, width), dtype=np.int)
    inp2 = np.zeros((height, width), dtype=np.int)
    inp3 = np.zeros((height, width), dtype=np.int)
    carrying = gameState.data.agentStates[index].numCarrying

    out = np.zeros((height, width, 3))
    mypos = gameState.data.agentStates[index].configuration.pos

    isRed = gameState.isOnRedTeam(index)
    numAgents = gameState.getNumAgents()
    out[int(mypos[1])][int(mypos[0])][2] = carrying + 1

    redFood = gameState.getRedFood()
    blueFood = gameState.getBlueFood()
    walls = gameState.getWalls()
    capsules = gameState.data.capsules
    for capsule in capsules:
        out[capsule[1]][capsule[0]][1] = 512
        out[capsule[1]][capsule[0]][1] = 512
    
    for x in range(height):
      for y in range(width):
        out[x][y][0] = -walls[y][x]
        if redFood[y][x]:
          out[x][y][1] = redFood[y][x]*256 if isRed else -redFood[y][x]*256
        if blueFood[y][x]:
          out[x][y][1] = -blueFood[y][x]*256 if isRed else blueFood[y][x]*256
    
    trans = False
    if not isRed:
      trans = True
      for i in range(out.shape[2]):
        self._trans(out[:,:,i])
    else:
      for i in range(out.shape[2]):
        out[:,:,i] = np.flip(out[:, :, i], 0)
    return np.expand_dims(out, axis = 0), trans

  def _reward(self, gameState, action):
    score_before = gameState.data.score
    gameState = gameState.generateSuccessor(self.index, self.actions[action])
    score_after = gameState.data.score
    redFoodEaten = self.numRedFood - len(gameState.getRedFood().asList())
    blueFoodEaten = self.numBlueFood - len(gameState.getBlueFood().asList())
    if gameState.isOnRedTeam(self.index):
      reward = blueFoodEaten - redFoodEaten + CaptureAgent.getScore(self,gameState)*100
    else:
      reward = redFoodEaten - blueFoodEaten + CaptureAgent.getScore(self,gameState)*100
    return reward

  def _trans_legal_actions(self, legal_actions):
    lst = []

    for x in legal_actions:
      lst.append(self.trans_actions_map[x])

    return lst
  def remove_illegal_actions(self, action_vector, legal_actions):
    '''
    Removes illegal actions
    '''
    action_vector = np.squeeze(action_vector)
    for i in range(len(action_vector)):
      if self.actions[i] not in legal_actions:
        action_vector[i] = np.min(action_vector)-1

    return action_vector

  def chooseAction(self, gameState):
    # Spá í að gera þetta eins og þetta var í byrjun.
    self.cnt+=1
    #if self.cnt%100 == 0:
      #print(self.cnt)
      #print(gameState)
    st, trans = self._convert_state_to_input(gameState, self.index)
    actions = gameState.getLegalActions(self.index)

    isRed = gameState.isOnRedTeam(self.index)
    #game = np.squeeze(st[:, :, :, 1] + st[:, :, :, 2])

   
    #print(isRed, trans, actions, a, self.trans_actions[a])
    if trans:
      actions = self._trans_legal_actions(actions)
    #print(actions)
    #print(gameState)
    #self.print_my_state(st)

    ac, self.terminal = self.model.sampleAction(self.policy, st, self.terminal)
    ac = self.remove_illegal_actions(ac, actions)
    #print("ac2", ac, self.index, self.policy)
    a = np.argmax(ac)

    action = self.trans_actions[a] if trans else a
    nextState = gameState.generateSuccessor(self.index, self.actions[action])
    nextState, trans = self._convert_state_to_input(nextState, self.index)
    reward = self._reward(gameState, action)
    
    ac = np.zeros((1, 4))
    ac[0][a] = 1

    if trans:
      a = self.trans_actions[a]
    #print("Reward", reward)
    return self.actions[a]

