from captureAgents import CaptureAgent
import distanceCalculator
import random
import time
import util
import sys
from game import Directions
import game
from util import nearestPoint

from miniMax import MiniMax, Node

def ScaredDefense(gameState, red, xDim, yDim, myPos, index, ennemy_index, getMazeDistance, getSuccessor, getSuccessorEnemy, evaluateScared):
    # Time for the minimax
    minimax_depth = 5
    minimax = MiniMax(gameState, None, minimax_depth)
    for layer in range(int((minimax_depth - 1)/2)):
        number_of_successors = 0
        for k in range(len(minimax.tree[2*layer])):
            node = minimax.tree[2*layer][k]
            actions = node.gameState.getLegalActions(index)
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
        ennemy_pos = terminalNode.gameState.getAgentPosition(ennemy_index)
        values = []
        # Compute objective point for this terminal node
        if red:
            xToCheck = xDim
        else:
            xToCheck = xDim - 1
        yToCheck = 0
        shortestDistance = 9999
        for y in range(1,yDim - 1):
            dist = getMazeDistance(myPos, ennemy_pos)
            if dist < shortestDistance:
                shortestDistance = dist
                yToCheck = y
        objPos = (xToCheck, yToCheck)
        posToGo = None
        for deltaX in range(-2, 3):
            for deltaY in range(-2, 3):
                posToCheck = (ennemy_pos[0] + deltaX, ennemy_pos[1] + deltaY)
                if posToCheck[0] > 0 and posToCheck[0] < xDim and posToCheck[1] > 0 and posToCheck[1] < yDim:
                    if gameState.data.layout.walls[posToCheck[0]][posToCheck[1]] == False:
                        if (getMazeDistance(posToCheck,objPos) == dist - 3) and (getMazeDistance(posToCheck, ennemy_pos) == 3):
                            posToGo = posToCheck
                            pass
        if posToGo == None:
            for deltaX in range(-2, 3):
                for deltaY in range(-2, 3):
                    posToCheck = (ennemy_pos[0] + deltaX, ennemy_pos[1] + deltaY)
                    if posToCheck[0] > 0 and posToCheck[0] < xDim and posToCheck[1] > 0 and posToCheck[1] < yDim:
                        if gameState.data.layout.walls[posToCheck[0]][posToCheck[1]] == False:
                            posToGo = posToCheck
                            pass
        if posToGo == None:
            posToGo = ennemy_pos
        # Find best action and give a value to this state
        for action in actions:
            values.append(evaluateScared(terminalNode.gameState, action, ennemy_pos, posToGo))
        terminalNode.value = max(values)
    # return best action to maximize the reward in 3 turns
    return minimax.ChooseBestAction()

def NOTONMYWATCH(gameState, diff, myPos, actions, getMazeDistance, getSuccessor, index, red, xDim):
    ennemy_pos = None
    if len(diff) == 1:
        ennemy_pos = diff[0]
        if ennemy_pos != None:
            current_dist = getMazeDistance(myPos, ennemy_pos)
            for action in actions:
                new_state = getSuccessor(gameState, action)
                new_dist = getMazeDistance(new_state.getAgentPosition(index), ennemy_pos)
                if new_dist < current_dist:
                    if (red and new_state.getAgentPosition(index)[0] < xDim - 1) or (not red and new_state.getAgentPosition(index)[0] > xDim):
                        return action, ennemy_pos
      # If two enemies are eating food, choose the enemy that is the closest the the center of the terrain
    if red:
        if len(diff) == 2:
            if diff[0][0] >= diff[1][0]:
                ennemy_pos = diff[0]
            else:
                ennemy_pos = diff[1]
    else:
        if len(diff) == 2:
            if diff[0][0] <= diff[1][0]:
                ennemy_pos = diff[0]
            else:
                ennemy_pos = diff[1]
    if ennemy_pos != None:
        current_dist = getMazeDistance(myPos, ennemy_pos)
        for action in actions:
            new_state = getSuccessor(gameState, action)
            new_dist = getMazeDistance(
                  new_state.getAgentPosition(index), ennemy_pos)
            if new_dist < current_dist:
                if (red and new_state.getAgentPosition(index)[0] < xDim - 1) or (not red and new_state.getAgentPosition(index)[0] > xDim):
                    return action, ennemy_pos
    return (None, None)