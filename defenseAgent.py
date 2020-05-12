from captureAgents import CaptureAgent
import distanceCalculator
import random
import time
import util
import sys
from game import Directions
import game
from util import nearestPoint
from math import *

from dangerMap import DangerMap
from attackSafeAgent import AttackSafeAgent

class DefenseAgent(CaptureAgent):
  """
  A base class for reflex agents that chooses score-maximizing actions
  """

  def __init__(self, index, timeForComputing=.1):
    # Agent index for querying state
    self.index = index
    # Whether or not you're on the red team
    self.red = None
    # Agent objects controlling you and your teammates
    self.agentsOnTeam = None
    # Maze distance calculator
    self.distancer = None
    # A history of observations
    self.observationHistory = []
    # Time to spend each turn on computing maze distances
    self.timeForComputing = timeForComputing
    # Access to the graphics
    self.display = None
    # The measured distances of each agent from
    self.distances = None

  def registerInitialState(self, gameState):
    self.start = gameState.getAgentPosition(self.index)
    CaptureAgent.registerInitialState(self, gameState)
    self.dangerMap = DangerMap(
        gameState.data.layout.walls, self.getMazeDistance)
    self.opponentsIndexes = self.getOpponents(gameState)
    self.teamIndexes = self.getTeam(gameState)

  def chooseAction(self, gameState):
    """
    Picks among the actions with the highest Q(s,a).
    """

    myPos = gameState.getAgentPosition(self.index)
    actions = gameState.getLegalActions(self.index)
    actions.remove('Stop')

    for agentIndex in self.opponentsIndexes:
      ennemy_pos = gameState.getAgentPosition(agentIndex)
      if ennemy_pos != None:
        current_dist = self.getMazeDistance(myPos, ennemy_pos)
        for action in actions:
          new_state = self.getSuccessor(gameState, action)
          new_dist = self.getMazeDistance(new_state.getAgentPosition(self.index),ennemy_pos)
          if new_dist < current_dist:
            if (self.red and new_state.getAgentPosition(self.index)[0] <= 14) or (not self.red and new_state.getAgentPosition(self.index)[0] >= 17):
              return action
    
    positions = self.bestPositions(gameState)
    
    if positions != [] and positions!= None:
      chosen_position = random.choice(positions)
      current_dist = self.getMazeDistance(myPos, chosen_position)
      for action in actions:
          new_state = self.getSuccessor(gameState, action)
          new_dist = self.getMazeDistance(new_state.getAgentPosition(self.index), chosen_position)
          if new_dist < current_dist:
              if (self.red and new_state.getAgentPosition(self.index)[0] <= 14) or (not self.red and new_state.getAgentPosition(self.index)[0] >= 17):
                  return action
    

    

    # You can profile your evaluation time by uncommenting these lines
    # start = time.time()
    values = [self.evaluate(gameState, a) for a in actions]
    # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]

    foodLeft = len(self.getFood(gameState).asList())

    if foodLeft <= 2:
      bestDist = 9999
      for action in actions:
        successor = self.getSuccessor(gameState, action)
        pos2 = successor.getAgentPosition(self.index)
        dist = self.getMazeDistance(self.start, pos2)
        if dist < bestDist:
          bestAction = action
          bestDist = dist
      return bestAction

    return random.choice(bestActions)

  def getSuccessor(self, gameState, action):
    """
    Finds the next successor which is a grid position (location tuple).
    """
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != nearestPoint(pos):
      # Only half a grid position was covered
      return successor.generateSuccessor(self.index, action)
    else:
      return successor

  def evaluate(self, gameState, action):
    """
    Computes a linear combination of features and feature weights
    """
    features = self.getFeatures(gameState, action)
    weights = self.getWeights(gameState, action)
    return features * weights

  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)

    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()

    # Computes whether we're on defense (1) or offense (0)
    features['onDefense'] = 1
    if myState.isPacman:
      features['onDefense'] = 0

    # Computes distance to invaders we can see
    enemies = [successor.getAgentState(i)
               for i in self.getOpponents(successor)]
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
    features['numInvaders'] = len(invaders)
    if len(invaders) > 0:
      dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
      features['invaderDistance'] = min(dists)

    if action == Directions.STOP:
      features['stop'] = 1
    rev = Directions.REVERSE[gameState.getAgentState(
        self.index).configuration.direction]
    if action == rev:
      features['reverse'] = 1

    return features

  def getWeights(self, gameState, action):
    return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2}

  def bestPositions(self, gameState):
    min_index = -1
    min_dist = 10000
    for enemyIndex in self.opponentsIndexes:
      dist = gameState.getAgentDistances()[enemyIndex]
      if dist < min_dist:
        min_dist = dist
        min_index = enemyIndex
    measured_distances = []
    indexes = []
    self.distances = gameState.getAgentDistances()[min_index]
    measured_distances.append(self.distances)
    indexes.append(self.index)
    measured_distances.append(AttackSafeAgent.distances[min_index])
    indexes.append(AttackSafeAgent.index)
    dis0 = self.possibleDistances(measured_distances[0])
    dis1 = self.possibleDistances(measured_distances[1])
    possiblepos = []
    for dist0 in dis0:
      for dist1 in dis1:
        myPos = gameState.getAgentState(self.index).getPosition()
        allyPos = gameState.getAgentState(indexes[1]).getPosition()
        pos = self.intersectionCircles(myPos[0], myPos[1], dist0, allyPos[0], allyPos[1], dist1)
        for position in pos:
          possiblepos.append(position)
    dictPos = {item:possiblepos.count(item) for item in possiblepos}
    if dictPos == {}:
      return None
    max_value = max(dictPos.values())
    best_pos = [key for key, value in dictPos.items() if value == max_value]
    return best_pos



  def intersectionCircles(self, xA, yA, r, xB, yB, R):
    """
    Returns the intersection points of the circle of center (xA, yA)
    and of redius r with the circle of center (xB,yB) and of radius
    R.
    Careful, we use the Manhattan distance for the circles !!!
    If there is no intersection, returns None
    """
    posA = self.circleManhattanCoords(xA, yA, r)
    posB = self.circleManhattanCoords(xB, yB, R)
    return [value for value in posA if value in posB]
  
  def possibleDistances(self, distance):
    res = []
    for k in range(-6,7):
      if distance + k > 5:
        res.append(distance + k)
    return res
  
  def circleManhattanCoords(self, x_pos, y_pos, r):
    res = []
    dangerMapMatrix = self.dangerMap.getDangerMap()
    if x_pos >= r and type(dangerMapMatrix[int(x_pos - r)][int(y_pos)]) == int:
      res.append((x_pos - r, y_pos))
    for x in range(-r + 1, r):
      y_plus = y_pos + r - abs(x_pos)
      y_minus = y_pos + abs(x_pos) - r
      if 0 < x_pos + x and x_pos + x < 32:
        if y_plus < 15 and 0 < y_plus and type(dangerMapMatrix[int(x_pos + x)][int(y_plus)]) == int:
          res.append((x_pos + x, y_plus))
        if 0 < y_minus and y_minus < 15 and type(dangerMapMatrix[int(x_pos + x)][int(y_minus)]) == int:
          res.append((x_pos + x, y_minus))
    if x_pos + r > 0 and x_pos + r < 31 and type(dangerMapMatrix[int(x_pos + r)][int(y_pos)]) == int:
      res.append((x_pos + r, y_pos))
    return res
