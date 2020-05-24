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
from copy import copy, deepcopy

class DangerMap1:

    def __init__(self, mapMatrix, getMazeDistance, xDim, yDim):
        self.initialDangerMap = mapMatrix
        self.dangerMap = mapMatrix
        self.getMazeDistance = getMazeDistance
        self.xDim = 2 * xDim
        self.yDim = yDim
        self.filledColumns = [False for x in range(xDim - 2)]
        # Setting walls to -1, middle to 1 and others to some big number
        for i in range(int(self.xDim/2)):
            for j in range(self.yDim):
                if self.initialDangerMap[i][j] == True:
                    self.dangerMap[i][j] = -1
                    self.dangerMap[self.xDim - i - 1][self.yDim - j - 1] = -1
                elif i == self.xDim/2 - 1:
                    self.dangerMap[i][j] = 1
                    self.dangerMap[self.xDim - i - 1][self.yDim - j - 1] = 1
                else:
                    self.dangerMap[i][j] = 5000
                    self.dangerMap[self.xDim - i - 1][self.yDim - j - 1] = 5000
        # Iterate until all cells are filled
        while not all(self.filledColumns):
            for i in range(int(self.xDim/2) - 2, 0, -1):
                if not self.filledColumns[i-1]:
                    for j in range(1, self.yDim - 1):
                        vals = self.returnNeighborsValues(i, j)
                        currentMin = self.dangerMap[i][j]
                        for val in vals:
                            if (val > 0) and (val < currentMin):
                                currentMin = val
                        if currentMin < self.dangerMap[i][j]:
                            self.dangerMap[i][j] = currentMin + 1
                            self.dangerMap[self.xDim - i - 1][self.yDim - j - 1] = currentMin + 1
                    column = self.dangerMap[i]
                    self.filledColumns[i-1] = not (5000 in column)
        self.initialDangerMap = self.dangerMap

    def getDangerMap(self):
        return self.dangerMap

    def getDanger(self, coords):
        return self.dangerMap[int(coords[0])][int(coords[1])]

    def resetDangerMap(self):
        self.dangerMap = self.initialDangerMap

    def addEnemy(self, enemyCoords):
        self.dangerMap = self.initialDangerMap
        for i in range(-5,6):
            for j in range(-5,6):
                ii = enemyCoords[0] + i
                jj = enemyCoords[1] + j
                if (ii >= 0) and (jj >= 0) and (ii < self.xDim / 2) and (jj < self.yDim) and (self.initialDangerMap[ii][jj] > -1):
                    dist = self.getMazeDistance((ii, jj), enemyCoords)
                    if dist < 6:
                        dist = 6 - dist
                        self.dangerMap[ii][jj] += 2*dist

    def addCapsule(self, capsuleCoords):
        for i in range(-2,3):
            for j in range(-2,3):
                ii = capsuleCoords[0] + i
                jj = capsuleCoords[1] + j
                if (ii >= 0) and (jj >= 0) and (ii < self.xDim / 2) and (jj < self.yDim) and (self.initialDangerMap[ii][jj] > -1):
                    dist = self.getMazeDistance((ii, jj), capsuleCoords)
                    if dist < 3:
                        self.dangerMap[ii][jj] = 0


    def returnNeighborsValues(self, i, j):
        vals = []
        addons = [[-1,0],[0,1],[1,0],[0,-1]]
        for addon in addons:
            a = addon[0]
            b = addon[1]
            ii = i + a
            jj = j + b
            if (ii >= 0) and (jj >= 0) and (ii < self.xDim/2) and (jj < self.yDim):
                vals.append(self.dangerMap[ii][jj])
            else:
                vals.append(-1)
        return vals
