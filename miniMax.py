from math import *
import numpy as numpy


class Node:

    def __init__(self, gameState, value, parent, parentAction, depth):
        self.gameState = gameState
        self.value = value
        self.parent = parent
        self.parentAction = parentAction
        self.depth = depth
        self.child = []

class MiniMax:

    def __init__(self, gameState, value, depth):
        initialNode = Node(gameState, value, None, None, depth)
        self.tree = [[initialNode]]
        for _ in range(depth - 1):
            self.tree.append([])
        self.total_depth = depth
    
    def ChooseBestAction(self):
        total_maxmin = (self.total_depth - 1) // 2 
        for k in range(total_maxmin, 0, -1):
            self.MinPart(2*k - 1)
            self.MaxPart(2*k - 2)
        best_value = -100000
        best_index = -1
        for k in range(len(self.tree[1])):
            node = self.tree[1][k]
            if node.value > best_value:
                best_value = node.value
                best_index = k
        return self.tree[1][best_index].parentAction
        
    def ChooseBestActionAlone(self):
        for k in range(len(self.tree) - 1, 0, -1):
            self.MaxPart(k - 1)
        best_value = -100000
        best_index = -1
        for k in range(len(self.tree[1])):
            node = self.tree[1][k]
            if node.value > best_value:
                best_value = node.value
                best_index = k
        print(best_value)
        return self.tree[1][best_index].parentAction

    def MaxPart(self, depth):
        for node in self.tree[depth]:
            values = []
            for childIndex in node.child:
                values.append(self.tree[depth + 1][childIndex].value)
            node.value = max(values)
    
    def MinPart(self, depth):
        for node in self.tree[depth]:
            values = []
            for childIndex in node.child:
                values.append(self.tree[depth + 1][childIndex].value)
            node.value = min(values)



    
    
