from captureAgents import CaptureAgent
import distanceCalculator
import random
import time
import util
import sys
from game import Directions
import game
from util import nearestPoint

from dangerMap1 import DangerMap1
from miniMax import MiniMax, Node

from attackFunctions import *
from defenseFunctions import *

class FlexibleAgent001(CaptureAgent):
    """
    A flexible agent capable of switching behaviors between attack and defense
    The 001 agent is initialized with attack properties
    """

    previous_pos = (0,0)
    distances = [10,10,10,10]
    index = 0
    previousFoodToDefend = [[0]]
    rememberedPosition = (0,0)
    currentPos = (0,0)

    attack = True

    def __init__(self, index, timeForComputing=.1):
        # Agent index for querying state
        FlexibleAgent001.attack = True
        self.index = index
        FlexibleAgent001.index = index
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
        # Dimensions of the map
        self.xDim = 16
        self.yDim = 16
        self.chaseCounter = 0

    def registerInitialState(self, gameState):
        self.start = gameState.getAgentPosition(self.index)
        CaptureAgent.registerInitialState(self, gameState)
        self.xDim = int(gameState.data.layout.walls.width/2)
        self.yDim = int(gameState.data.layout.walls.height)
        self.dangerMap = DangerMap1(gameState.data.layout.walls, self.getMazeDistance, self.xDim, self.yDim)
        #print(self.dangerMap.getDangerMap())
        self.opponentsIndexes = self.getOpponents(gameState)
        self.initialFoodLeft = len(self.getFood(gameState).asList())
        self.k = 0
        self.teamIndexes = self.getTeam(gameState)
        FlexibleAgent001.previousFoodToDefend = self.getFoodYouAreDefending(gameState)
    
    def chooseAction(self, gameState):
        """
        Picks the best action to do, depending on current behavior
        """

        # Update measurements
        FlexibleAgent001.distances = gameState.getAgentDistances()
        FlexibleAgent001.currentPos = gameState.getAgentPosition(self.index)

        # Attack behavior
        if FlexibleAgent001.attack:
            FlexibleAgent001.previousFoodToDefend = self.getFoodYouAreDefending(gameState)
            # Find the closest enemy and proceed safely if required
            closeEnemyPositions = []
            closeEnemyIndex = []
            closeEnemyDistances = []
            myPos = gameState.getAgentPosition(self.index)
            for ennemy_index in self.opponentsIndexes:
                ennemy_pos = gameState.getAgentPosition(ennemy_index)
                if ennemy_pos != None:
                    closeEnemyPositions.append(ennemy_index)
                    closeEnemyDistances.append(self.getMazeDistance(ennemy_pos, myPos))
                    closeEnemyIndex.append(ennemy_index)
                if closeEnemyIndex != []:
                    # An ennemy is near ! Be safe !
                    return AttackSafe(gameState, self.index, ennemy_index, self.getSuccessor, self.getSuccessorEnemy, self.evaluateAttack)
            
            # Check how much food the agent carries and order it to go back if he carries too much
            numCarrying = gameState.getAgentState(self.index).numCarrying
            myPos = gameState.getAgentPosition(self.index)
            if numCarrying >= int(self.initialFoodLeft/5):
                return FGoBack(gameState, self.xDim, self.yDim, self.red, myPos, self.getMazeDistance, self.index, self.getSuccessor)

            # If no other case was triggered, you're safe and have some work to do! Just go collect some food
            foodLeft = len(self.getFood(gameState).asList())
            return CollectFoodForTheWin(gameState,
                                        self.index,
                                        self.evaluateAttack,
                                        self.getSuccessor,
                                        self.getFood,
                                        self.getFeaturesAttack,
                                        self.getWeightsAttack,
                                        foodLeft,
                                        self.getMazeDistance,
                                        self.start)
        
        # Defense behavior
        if not FlexibleAgent001.attack:
            myPos = gameState.getAgentPosition(self.index)
            actions = gameState.getLegalActions(self.index)
            actions.remove('Stop')
            scared = gameState.getAgentState(self.index).scaredTimer > 0

            # If you are defending and scared, you stay at a reasonable distance from the opponent while trying to prevent him from going to its side of the terrain
            if scared:
                for ennemy_index in self.opponentsIndexes:
                    ennemy_pos = gameState.getAgentPosition(ennemy_index)
                if ennemy_pos != None:
                    return ScaredDefense(gameState,
                                         self.red,
                                         self.xDim,
                                         self.yDim,
                                         myPos,
                                         self.index,
                                         ennemy_index,
                                         self.getMazeDistance,
                                         self.getSuccessor,
                                         self.getSuccessorEnemy,
                                         self.evaluateScared)
            
            # If you are not scared and that some food is being eaten in your town, you should check there
            currentFoodToDefend = self.getFoodYouAreDefending(gameState)
            diff = []
            for x in range(self.xDim):
                for y in range(self.yDim):
                    if currentFoodToDefend[x][y] != FlexibleAgent001.previousFoodToDefend[x][y]:
                        diff.append((x,y))
            FlexibleAgent001.previousFoodToDefend = currentFoodToDefend
            if diff != []:
                actionToTake, posToGo = NOTONMYWATCH(gameState,
                                                     diff,
                                                     myPos,
                                                     actions,
                                                     self.getMazeDistance,
                                                     self.getSuccessor,
                                                     self.index,
                                                     self.red,
                                                     self.xDim)
                if actionToTake != None:
                    self.chaseCounter = 7
                    FlexibleAgent001.rememberedPosition = (int(posToGo[0]), int(posToGo[1]))
                    return actionToTake
            
            # If an ennemy is on your side, and you can detect it, you should chase it !
            for agentIndex in self.opponentsIndexes:
                ennemy_pos = gameState.getAgentPosition(agentIndex)
                if ennemy_pos != None and not scared:
                    current_dist = self.getMazeDistance(myPos, ennemy_pos)
                    for action in actions:
                        new_state = self.getSuccessor(gameState, action)
                        new_dist = self.getMazeDistance(new_state.getAgentPosition(self.index),ennemy_pos)
                        if new_dist < current_dist:
                            if (self.red and new_state.getAgentPosition(self.index)[0] < self.xDim - 1) or (not self.red and new_state.getAgentPosition(self.index)[0] > self.xDim):
                                return action
            
            # If you can't see any enemy and you found out the enemy was somewhere, go to the remembered position
            myPos = gameState.getAgentPosition(self.index)
            ennemy_pos = FlexibleAgent001.rememberedPosition
            if self.chaseCounter > 0 and ennemy_pos != (0,0):
                self.chaseCounter -= 1
                current_dist = self.getMazeDistance(myPos, ennemy_pos)
                for action in actions:
                    new_state = self.getSuccessor(gameState, action)
                    new_dist = self.getMazeDistance(
                        new_state.getAgentPosition(self.index), ennemy_pos)
                    if new_dist < current_dist:
                        if (self.red and new_state.getAgentPosition(self.index)[0] < self.xDim - 1) or (not self.red and new_state.getAgentPosition(self.index)[0] > self.xDim):
                            return action
            
            # If no case was correct, maybe you should check the closest food to the enemy
            objectivePosition = self.closestFoodToEnemy(gameState)
            current_dist = self.getMazeDistance(myPos, objectivePosition)
            for action in actions:
                new_state = self.getSuccessor(gameState, action)
                new_dist = self.getMazeDistance(
                        new_state.getAgentPosition(self.index), objectivePosition)
                if new_dist < current_dist:
                    if (self.red and new_state.getAgentPosition(self.index)[0] < self.xDim - 1) or (not self.red and new_state.getAgentPosition(self.index)[0] > self.xDim):
                        return action
            
            # If nothing was done yet, just stay there, staring into the void as you're confused
            return 'Stop'

    def closestFoodToEnemy(self, gameState):
        """
        Returns the position of the food that is closest to the enemy spawn
        """
        foodToDefend = self.getFoodYouAreDefending(gameState)
        if self.red:
            min_dist = 9999
            min_pos = (0,0)
            for x in range(0, self.xDim - 1):
                for y in range(1, self.yDim - 1):
                    if foodToDefend[x][y] == True:
                        dist = self.getMazeDistance((x, y), (2*self.xDim - 2, self.yDim - 2))
                        if dist < min_dist:
                            min_dist = dist
                            min_pos = (x,y)
        else:
            min_dist = 9999
            min_pos = (0,0)
            for x in range(1, self.xDim):
                for y in range(1, self.yDim - 1):
                    if foodToDefend[x][y] == True:
                        dist = self.getMazeDistance((x, y), (1,1))
                        if dist < min_dist:
                            min_dist = dist
                            min_pos = (x,y)
        return min_pos




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
  
    def getSuccessorEnemy(self, gameState, action, enemyIndex):
        successor = gameState.generateSuccessor(enemyIndex, action)
        pos = successor.getAgentState(enemyIndex).getPosition()
        if pos != nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generateSuccessor(enemyIndex, action)
        else:
            return successor
    
    
    def evaluateAttack(self, gameState, action):
        """
        Computes a linear combination of features and feature weights when in the attack stance
        """
        features = self.getFeaturesAttack(gameState, action)
        weights = self.getWeightsAttack(gameState, action)
        state_reward = features * weights
        numCarrying = gameState.getAgentState(self.index).numCarrying

        successor = self.getSuccessor(gameState, action)
        futurePos = successor.getAgentState(self.index).getPosition()
        myPos = gameState.getAgentPosition(self.index)
        deposit_reward = 0
        if self.red:
            if myPos[0] == self.xDim and futurePos[0] == self.xDim - 1:
                deposit_reward += numCarrying*100
        else:
            if myPos[0] == self.xDim - 1 and futurePos[0] == self.xDim:
                deposit_reward += numCarrying*100

        return state_reward + deposit_reward

    def getFeaturesAttack(self, gameState, action):
        """
        Returns the features when in attack stance
        """
        
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        myPos = successor.getAgentState(self.index).getPosition()
        foodList = self.getFood(successor).asList()
        features['successorScore'] = -len(foodList)

        # Compute distance to the nearest food
        if len(foodList) > 0:  # This should always be True,  but better safe than sorry
            minDistance = min([self.getMazeDistance(myPos, food)
                                for food in foodList])
            features['distanceToFood'] = minDistance
        
        # Compute danger at agent position
        if (myPos[0] <= self.xDim - 1 and self.red) or (myPos[0] >= self.xDim and not self.red):
            features['danger'] = 0
        else:
            features['danger'] = self.dangerMap.getDanger(myPos)

        # Compute a metric representing how close is the closest ennemy
        distances = gameState.getAgentDistances()
        FlexibleAgent001.distances = gameState.getAgentDistances()
        min_dist = 100
        scared = False
        for ennemy_index in self.opponentsIndexes:
            ennemy_pos = gameState.getAgentPosition(ennemy_index)
            if ennemy_pos != None:
                dist = self.getMazeDistance(ennemy_pos,myPos)
            else:
                dist = max(6,distances[ennemy_index])
            if dist < min_dist:
                min_dist = dist
            if gameState.getAgentState(ennemy_index).scaredTimer > 2:
                scared = True

        if min_dist > self.xDim - 6:
            features['ennemyProximity'] = 1000000   
        elif min_dist == 5 or (myPos[0] < self.xDim and self.red) or (myPos[0] >= self.xDim and not self.red) or scared:
            features['ennemyProximity'] = 0
        elif min_dist > 4:
            features['ennemyProximity'] = 20
        elif min_dist > 3:
            features['ennemyProximity'] = 50
        elif min_dist > 2:
            features['ennemyProximity'] = 100
        elif min_dist > 1:
            features['ennemyProximity'] = 200
        elif min_dist > 0:
            features['ennemyProximity'] = 5000
        elif min_dist <= 0:
            features['ennemyProximity'] = 1000000

        capsules = self.getCapsules(gameState)
        min_dist = 9999
        for capsule in capsules:
            dist = self.getMazeDistance(capsule, myPos)
            if dist < min_dist:
                min_dist = dist
        if min_dist > 3:
            features['capsuleProximity'] = 0
        elif min_dist > 2:
            features['capsuleProximity'] = 150
        elif min_dist > 1:
            features['capsuleProximity'] = 300
        elif min_dist > 0:
            features['capsuleProximity'] = 6000
        elif min_dist <= 0:
            features['capsuleProximity'] = 1000000

        enemies = [successor.getAgentState(i)
                for i in self.getOpponents(successor)]
        invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
        features['numInvaders'] = len(invaders)

        return features

    def getWeightsAttack(self, gameState, action):
        return {'successorScore': 10, 'distanceToFood': -3, 'danger': -0.0, 'ennemyProximity': -2, 'capsuleProximity': 2, 'numInvaders': -100}
    
    def evaluateDefense(self, gameState, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.getFeaturesDefense(gameState, action)
        weights = self.getWeightsDefense(gameState, action)
        return features * weights

    def evaluateScared(self, gameState, action, ennemy_pos, posToGo):
        """
        Gives the score to a gamestate when the agent is scared
        """
        myPos = gameState.getAgentPosition(self.index)
        distToEnnemy = self.getMazeDistance(myPos, ennemy_pos)
        distToObj = self.getMazeDistance(myPos, posToGo)
        return 0.8*distToEnnemy - distToObj

    def getFeaturesDefense(self, gameState, action):
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

    def getWeightsDefense(self, gameState, action):
        return {'numInvaders': -2000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2}