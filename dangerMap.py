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

class DangerMap:

  def __init__(self, mapMatrix, getMazeDistance, xDim, yDim):
    self.initialDangerMap = mapMatrix
    self.dangerMap = mapMatrix
    self.getMazeDistance = getMazeDistance
    self.xDim = xDim
    self.yDim = yDim
    for x in range(1, self.xDim - 1):
      for y in range(1, self.yDim - 1):
        if self.initialDangerMap[x][y] == False:
          self.initialDangerMap[x][y] = 500
          self.initialDangerMap[2*self.xDim - 1 -x][self.yDim - 1 -y] = 500
    # Initializing the middle with 0s
    for y in range(1, self.yDim - 1):
      if self.initialDangerMap[self.xDim - 1][y] == False:
          self.initialDangerMap[self.xDim - 1][y] = 0
          self.initialDangerMap[self.xDim][self.yDim - 1 -y] = 0
    # Construct iteratively the map
    for _ in range(1):
      for x in range(self.xDim - 2,0,-1):
        for y in range(1, self.yDim - 1):
          if type(self.dangerMap[x][y]) == int:
            close_positions = self.returnCorrectNeighbours(x,y)
            limit_positions = self.returnLimitsCoordinates(x,y)
            min_danger_on_close_positions = 500
            for coord in close_positions:
              danger = self.dangerMap[coord[0]][coord[1]]
              if coord[1] >= self.xDim:
                  danger = 0
              max_possible_danger = 0
              for ennemy_coord in limit_positions:
                diff = self.getMazeDistance((x, y), coord) - self.getMazeDistance(ennemy_coord, coord)
                if diff >= 0:
                  if max_possible_danger < danger + diff:
                    max_possible_danger = danger + diff
                else:
                  if max_possible_danger < danger:
                    max_possible_danger = danger
              if max_possible_danger < min_danger_on_close_positions:
                min_danger_on_close_positions = max_possible_danger
            self.dangerMap[x][y] = min_danger_on_close_positions
            self.dangerMap[2*self.xDim - 1 -x][self.xDim - 1 -y] = min_danger_on_close_positions
    self.initialDangerMap = deepcopy(self.dangerMap)
                

    
  
  def getDangerMap(self):
    return self.dangerMap
  
  def getDanger(self, coords):
    return self.dangerMap[int(coords[0])][int(coords[1])]
  
  def returnCorrectNeighbours(self, x_pos, y_pos):
    res = []
    for x in range(-5, 6):
      for y in range(abs(x) - 5, 6 - abs(x)):
        if 0 < x_pos + x and x_pos + x < 2*self.xDim - 1 and 0 < x_pos and x_pos < 2*self.xDim - 1:
          if 0 < y_pos + y and y_pos + y < self.yDim - 1 and 0 < y_pos and y_pos < self.yDim - 1:
            if self.initialDangerMap[x_pos + x][y_pos + y] == False and self.initialDangerMap[x_pos][y_pos] == False:
              print(self.dangerMap)
              if self.getMazeDistance((x_pos,y_pos),(x_pos + x,y_pos + y)) <= 5:
                res.append((x_pos + x,y_pos + y))
    return res
  
  def returnLimitsCoordinates(self, x_pos, y_pos):
    res = []
    if x_pos >= 6 and type(self.dangerMap[x_pos - 5][y_pos]) == int:
      res.append((x_pos - 5, y_pos))
    for x in range(-4, 5):
      y_plus = y_pos + 5 - abs(x_pos)
      y_minus = y_pos + abs(x_pos) - 5
      if 0 < x_pos + x:
        if y_plus < self.xDim - 1 and 0 < y_plus and type(self.dangerMap[x_pos + x][y_plus]) == int:
          res.append((x_pos + x,y_plus))
        if 0 < y_minus and y_minus < self.yDim - 1 and type(self.dangerMap[x_pos + x][y_minus]) == int:
          res.append((x_pos + x, y_minus))
    if type(self.dangerMap[x_pos + 5][y_pos]) == int:
      res.append((x_pos + 5, y_pos))
    return res



  

