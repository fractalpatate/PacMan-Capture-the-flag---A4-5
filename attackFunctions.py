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

# from agent1 import FlexibleAgent001
# from agent2 import FlexibleAgent002

def AttackSafe(gameState, index, ennemy_index, getSuccessor, getSuccessorEnemy, evaluate):
    """
    When using this, the attack agent has found some ennemy close and chose the closest to him. He then use
    minimax to continue to collect food while staying safe (going to the best evaluated state).

    Inputs :
        gameState : current gameState
        index : the index of the agent
        ennemy_index : the index of the closest opponent
        getSuccessor : the function that computes the next gameState knowing which action will be done by our agent
        getSuccessorEnemy : the function that computes the next gameState knowing which action will be done by the opponent
        evaluate : the function used to evaluate each state
    """
    # Minimax depth must be odd
    minimax_depth = 5
    minimax = MiniMax(gameState, None, minimax_depth)
    for layer in range(int((minimax_depth - 1)/2)):
        number_of_successors = 0
        for k in range(len(minimax.tree[2*layer])):
            node = minimax.tree[2*layer][k]
            actions = node.gameState.getLegalActions(index)
            actions.remove('Stop')
            for action in actions:
                successor = getSuccessor(node.gameState, action)
                minimax.tree[2*layer + 1].append(Node(successor, 0, k, action, node.depth - 1))
                minimax.tree[2*layer][k].child.append(number_of_successors)
                number_of_successors += 1
        number_of_successors = 0
        for k in range(len(minimax.tree[2*layer + 1])):
            node = minimax.tree[2*layer + 1][k]
            actions = node.gameState.getLegalActions(ennemy_index)
            for action in actions:
                successor = getSuccessorEnemy(node.gameState, action, ennemy_index)
                minimax.tree[2*layer + 2].append(Node(successor, 0, k, action, node.depth - 1))
                minimax.tree[2*layer + 1][k].child.append(number_of_successors)
                number_of_successors += 1
    for terminalNode in minimax.tree[len(minimax.tree) - 1]:
        actions = terminalNode.gameState.getLegalActions(index)
        values = []
        for action in actions:
            values.append(evaluate(terminalNode.gameState, action))
            terminalNode.value = max(values)
    return minimax.ChooseBestAction()    

def FGoBack(gameState, xDim, yDim, red, myPos, getMazeDistance, index, getSuccessor):
    """
    When using this function, the agent just tries to go back to its side of the terrain using the closest path.
    """
    min_dist = 9999
    if red:
        xToGo = xDim - 1
    else:
        xToGo = xDim
    for yCoord in range(1,yDim):
        if gameState.data.layout.walls[xToGo][yCoord] == False:
            dist = getMazeDistance((xToGo, yCoord), myPos)
            if dist < min_dist:
                min_dist = dist
                min_coord = (xToGo, yCoord)
    actions = gameState.getLegalActions(index)
    actions.remove('Stop')
    values = [getMazeDistance(min_coord, getSuccessor(gameState, action).getAgentPosition(index)) for action in actions]
    minValue = min(values)
    bestActions = [a for a, v in zip(actions, values) if v == minValue]
    return random.choice(bestActions)

def CollectFoodForTheWin(gameState, index, evaluate, getSuccessor, getFood, getFeatures, getWeights, foodLeft, getMazeDistance, start):
    """
    When using this function, the agent only tries to maximize its reward one step ahead in time. (the evaluate function is important)
    Be sure to you it when the agent is safe !
    """
    # We promote movement
    actions = gameState.getLegalActions(index)
    actions.remove('Stop')
    values = []
    for action in actions:
        value = evaluate(gameState, action)
        successor_pos = getSuccessor(gameState, action).getAgentPosition(index)
        if getFood(gameState)[successor_pos[0]][successor_pos[1]]:
            features = getFeatures(gameState, action)
            weights = getWeights(gameState, action)
            value -= features['distanceToFood']*weights['distanceToFood']
        
        values.append(value)
    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]
    # Go back if the winning condition are about to be met
    if foodLeft <= 2:
        bestDist = 9999
        for action in actions:
            successor = getSuccessor(gameState, action)
            pos2 = successor.getAgentPosition(index)
            dist = getMazeDistance(start, pos2)
            if dist < bestDist:
                bestAction = action
                bestDist = dist
        return bestAction

    return random.choice(bestActions)