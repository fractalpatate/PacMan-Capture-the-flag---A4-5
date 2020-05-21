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
import numpy as np

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
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """

  # The following line is an example only; feel free to change it.
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########


class DummyAgent(CaptureAgent):
  """
  A Dummy agent to serve as an example of the necessary agent structure.
  You should look at baselineTeam.py for more details about how to
  create an agent as this is the bare minimum.
  """

  agentObjects=dict()

  #def __init__(self,isRed, **args):
  #  super.__init__(isRed, **args)
  #  self.opponent_indices = []

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
    self.numberOfAgents = gameState.getNumAgents()
    self.opponent_starting_pos = []
    self.opponent_indeces = self.getOpponents(gameState)
    for enemy in self.opponent_indeces:
        self.opponent_starting_pos.append(gameState.getInitialAgentPosition(enemy))
    # print(str(self.opponent_starting_pos), flush=True)
    # print('Bad boys indeces are ', self.opponent_indeces, flush=True)
    # print('I am in team red?', self.red,flush=True)
    self.turn=0
    self.numberOfParticles=100
    self.particles=[]
    for i in range(int(self.numberOfAgents/2)):
        self.particles.append([])
    # print('I am player ',self.index,flush=True)
    # print('size of our empty list',len(self.particles),flush=True)
    # print('our empty list',self.particles,flush=True)
    self.getTeammateIndex(gameState)
    # print('agents in my team',self.agentsOnTeam,flush=True)

    self.isOnRedTeam = gameState.isOnRedTeam(self.index)
    self.prev_gamestate_food = self.getFoodYouAreDefending(gameState)
    self.foodWithin=0
    self.populateHalfPoints(gameState)
    DummyAgent.agentObjects[self.index]=self
    self.opponents_position_distribution = None
    if self.index<self.otherAgentIndex:
        self.offensive=True
    else:
        self.offensive=False
    self.scared=0


  def inspectEnvironment(self,gameState):
    gameState = gameState.deepCopy()
    if self.turn==0: # if it is the first turn
        for opponent in self.opponent_indeces:
            starting_position = list(gameState.getInitialAgentPosition(opponent))[:]
            self.populatePosition(starting_position,opponent)

    my_index = self.index
    y_observation = gameState.getAgentDistances()
    for opponent_index in self.opponent_indeces:
        opponent_distance = y_observation[opponent_index]

        if not gameState.getAgentPosition(opponent_index): # if we cannot observe the opponent
            hasJustMoved = (opponent_index == (self.index-1)%self.numberOfAgents)
            someoneAte, positionOfEating = self.inspectOurFood(gameState)
            if hasJustMoved and someoneAte:
                # print('just ate yo',positionOfEating[:],flush=True)
                self.populatePosition(positionOfEating[:],opponent_index)
            else:
                particle_list = self.particles[opponent_index//2][:]
                self.particles[opponent_index//2] = self.estimateOpponentPositionAlternative(gameState,opponent_index,opponent_distance,hasJustMoved, particle_list)
        else:
            opponent_position = gameState.getAgentPosition(opponent_index)
            self.populatePosition(list(opponent_position)[:],opponent_index)

  def inspectOurFood(self,gameState):
      new_food_matrix = self.getFoodYouAreDefending(gameState)
      for i in range(new_food_matrix.width):
          for j in range(new_food_matrix.height):
              if new_food_matrix[i][j] == False and self.prev_gamestate_food[i][j] == True:
                return True, list([i,j])

      return False, None

  def populatePosition(self, pos,index):
    particles_for_specific_enemy = []
    for i in range(self.numberOfParticles):
        w = 1/self.numberOfParticles
        particles_for_specific_enemy.append([pos,w])
    self.particles[index//2] = particles_for_specific_enemy

  def getTeammateIndex(self, gameState):
      our_index = self.index
      our_team_indeces=  self.getTeam(gameState)
      for index in our_team_indeces:
          if index != our_index:
              self.otherAgentIndex=index

  def updateParticle(self, gameState,index, isMoving,previousPos):
      gameState = gameState.deepCopy()
      if isMoving==False:
          return previousPos
      else:
        possible_blocks = self.get_open_cells(gameState,previousPos)
        next_pos = []
        next_pos = random.choice(possible_blocks)[:]
        return next_pos

  def replace_at_index(self,tup, ix, val):
    lst = list(tup)
    lst[ix] = val
    return tuple(lst)


  def estimateOpponentPositionAlternative(self, gameState, opponent_index, opponent_distance,hasJustMoved, particle_list):
    if hasJustMoved==False:
        allyAgent = DummyAgent.agentObjects[self.otherAgentIndex]
        allyEnemyParticles = allyAgent.particles[opponent_index//2]
        return allyEnemyParticles
    else:
        gameStateCopy = gameState.deepCopy()
        eta = 0 # normalisation parameter
        # update weights
        # print('Turn ',self.turn,flush=True)
        # print('I have Player number',self.index,flush=True)
        # print('I am estimating the position of opponent_index:', opponent_index,flush=True)
        weight_array = np.zeros(len(particle_list))
        #print('My position at estimateOpponent is at',gameState.getAgentPosition(self.index),flush=True)
        #print('particle list before', particle_list,flush=True)
        counter=0
        # copy the past and merge particles located in the same spot
        new_particles = []

        for i in range(len(particle_list)):
            particle = particle_list[i]
            position = particle[0]
            weight = particle[1]
            alreadyExists = False
            for j in range(len(new_particles)):
                new_particle = new_particles[j]
                if new_particle[0] == position:
                    new_particle[1] +=weight
                    alreadyExists = True
            if not alreadyExists:
                new_particles.append(list([position,weight]))

        size_of_past = len(new_particles)
        weights = np.zeros(size_of_past)
        for i in range(size_of_past):
            weights[i] = new_particles[i][1]

        if hasJustMoved==True:
            for i in range(len(new_particles)):
                location = new_particles[i][0]
                weight = new_particles[i][1]
                next_locations = self.get_open_cells(gameState,location)
                for loc in next_locations:
                    new_particles.append(list([loc,weight]))

        size_thus_far = len(new_particles)
        if size_thus_far< self.numberOfParticles:
            indeces = np.random.choice(np.arange(size_of_past), size=self.numberOfParticles - size_thus_far,p=weights,replace=True)
            for index in indeces:
                if hasJustMoved:
                    prev_location =  new_particles[index][0][:]
                    next_locations = self.get_open_cells(gameState,prev_location)
                    next_loc = random.choice(next_locations)
                    weight = new_particles[index][1]
                    new_particles.append(list([next_loc,weight]))
                else:
                    location = new_particles[index][0][:]
                    weight = new_particles[index][1]
                    new_particles.append(list([location,weight]))


        for i in range(len(new_particles)):
            position = new_particles[i][0]
            weight = new_particles[i][1]
            trueDistance = int(util.manhattanDistance(gameState.getAgentPosition(self.index), tuple(position)))
            weight = weight * gameState.getDistanceProb(trueDistance,opponent_distance)
            new_particles[i][1] = weight
            eta += weight
        # print('Eta sum', eta,flush=True)
        for i in range(len(new_particles)):
            new_particles[i][1] = new_particles[i][1]/eta
            #print('Weight of new particles',new_particles[i][1],flush=True)
        return new_particles


  
  def inOurHalf(self,gameState):
    foodMatrix = self.getFood(gameState)
    x_dimension = foodMatrix.width

    current_position = gameState.getAgentPosition(self.index)
    if self.isOnRedTeam:
        return current_position[0] < int(x_dimension/2)
    else:
        return current_position[0] >= int(x_dimension/2) # todo will have to change for other team

    # now we resample based on weights
  def generateDistribution(self, particle_list):
    distributions = []              # We have one distribution per agent
    distribution = util.Counter()   # Creating a new counter for one agent
    for j in range(len(particle_list)):
        location = particle_list[j][0]
        weight = particle_list[j][1]

        distribution[tuple(location)]+=weight

    distributions.append(distribution)
    for counters in list(distribution.keys()):
        location_list = list(counters)
        weight = distribution[counters]
    #print('Does the distribution sum up to ?', prob, flush=True)

    return distributions

  def get_open_cells(self, gameState, pos):
    """
    Looks around the input position on each adjustance square in manhattan format and returns list of cells that are open (not walls)
    """
    gameState = gameState.deepCopy()
    open_cells = [pos]
    try:
        if not gameState.hasWall(pos[0]+1, pos[1]):
            open_cells.append([pos[0]+1, pos[1]])
        if not gameState.hasWall(pos[0]-1, pos[1]):
            open_cells.append([pos[0]-1, pos[1]])
        if not gameState.hasWall(pos[0], pos[1]+1):
            open_cells.append([pos[0], pos[1]+1])
        if not gameState.hasWall(pos[0], pos[1]-1):
            open_cells.append([pos[0], pos[1]-1])
    except:
            print("Went outside index, i expled but it is okay i am in a try-catch block", flush=True)
    return open_cells


  def chooseAction(self, gameState):
    """
    Picks among actions randomly.
    """
    self.scared = (gameState.data.agentStates[self.index].scaredTimer>0)

    if self.inOurHalf(gameState):
        self.foodWithin=0

    #print('Game Object list size',len(DummyAgent.agentObjects))
    self.gameState=gameState.deepCopy()
    actions = gameState.getLegalActions(self.index)

   
    previous_action = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
    # print('Previous Action', previous_action,flush=True)
    if self.offensive:
        actions.remove('Stop')
        if len(actions)>1 and previous_action in actions:
            actions.remove(previous_action)

    previous_game_state = self.getPreviousObservation()
    if previous_game_state is not None:
        #self.prev_gamestate_food = self.getFoodYouAreDefending(previous_game_state)
        allyAgent = DummyAgent.agentObjects[self.otherAgentIndex]
        self.prev_gamestate_food = allyAgent.prev_gamestate_food.deepCopy()
    opponent_suicided, opponent_index = self.opponentSuicided(previous_game_state,gameState)

    if opponent_suicided:
        allyAgent = DummyAgent.agentObjects[self.otherAgentIndex]
        starting_position = list(gameState.getInitialAgentPosition(opponent_index))[:]
        self.populatePosition(starting_position,opponent_index)
        allyAgent.populatePosition(starting_position,opponent_index)
    self.inspectEnvironment(gameState)
    

    new_distribution = self.generateDistribution(self.particles[0])
    new_distribution = new_distribution + self.generateDistribution(self.particles[1])
    self.opponents_position_distribution = new_distribution
    self.displayDistributionsOverPositions(new_distribution)
    self.prev_gamestate_food = self.getFoodYouAreDefending(gameState)
    '''
    You should change this in your own agent.
    '''
    best_utility = -np.inf
    best_action = None
    best_next_gameState = None
    midPoint = int(gameState.data.layout.width/2)
    for action in actions:
        # print('About to perform this action', action,flush=True)
        nextGameState = gameState.generateSuccessor(self.index,action)


        if self.offensive == True:
            utility = self.getOffensiveUtilityOfGameState(nextGameState)
        else:
            ourNextPosition = list(nextGameState.getAgentPosition(self.index))
            if self.isOnRedTeam==False and ourNextPosition[0]<midPoint:
                continue
            if self.isOnRedTeam==True and ourNextPosition[0] >=midPoint:
                continue
            utility = self.getDefensiveUtilityOfGameState(nextGameState)
        if best_utility< utility:
            best_utility = utility
            best_action = action
            best_next_gameState = nextGameState
    # print('best utility score', best_utility,flush=True)
#    print('I have this much food in me', self.foodWithin,flush=True)
    if self.agentAte(gameState,best_next_gameState):
        self.foodWithin += 1

    # print('best utility function', best_utility,flush=True)
    ateOpponent, indexOfEatenEnemy = self.weEatOpponent(gameState,best_action)
    if ateOpponent:     # Reset enemy position after eating it
#       print("\n\nI just triggered yo",flush=True)
        starting_position = list(gameState.getInitialAgentPosition(indexOfEatenEnemy))[:]
        for agents in DummyAgent.agentObjects.values():
#            print('updating agent',agents.index,flush=True)
            agents.populatePosition(starting_position,indexOfEatenEnemy)
#            print("He respawned at position",starting_position,flush=True)

    self.turn+=1
    return best_action
    #return random.choice(actions)
  
  def opponentSuicided(self,previous_game_state,gameState):
    if self.turn==0:
        return False,0
    enemy_locations =  [self.opponents_position_distribution[i].argMax() for i in range(len(self.opponents_position_distribution))]
    ourPosition = gameState.getAgentPosition(self.index)
    for enemy in self.opponent_indeces:
        enemy_agent_prev = previous_game_state.data.agentStates[enemy]
        enemy_agent = gameState.data.agentStates[enemy]
        if gameState.getAgentPosition(enemy) is None:
            if previous_game_state.getAgentPosition(enemy) is not None:
                exact_location = enemy_locations[enemy//2]
                # print('exact_location', exact_location,flush=True)
                if self.getMazeDistance(tuple(exact_location),ourPosition)==1:
                    return True,enemy
    return False,0

  def weEatOpponent(self,gameState,action):
    y_observation = gameState.getAgentDistances()
    nextState = gameState.generateSuccessor(self.index, action)
    for enemies in self.opponent_indeces:
        opponent_distance_before = y_observation[enemies]
        if gameState.getAgentPosition(enemies) is not None:
            opponent_position = gameState.getAgentPosition(enemies)
            my_position = gameState.getAgentPosition(self.index)
            opponent_distance_before =  self.getMazeDistance(my_position,opponent_position)
            next_position = nextState.getAgentPosition(self.index)
 #           print('My position is at', my_position,flush=True)
 #           print('Enemy is at', opponent_position, flush=True)
 #           print('I plan to go to', next_position, flush=True)
            if opponent_distance_before==1 and next_position == opponent_position:
                enemy_state = gameState.data.agentStates[enemies]
                if enemy_state.isPacman or enemy_state.scaredTimer>0:
                    # print('I am about to eat a pacman', flush=True)
                    return True, enemies

    return False, 0

  def agentAte(self,gameState,nextGameState):
    currentFood = self.getFood(gameState)
    foodAfterAction = self.getFood(nextGameState)
    for i in range(currentFood.width):
        for j in range(currentFood.height):
            if currentFood[i][j] != foodAfterAction[i][j]:
                return True
    return False
      

  def getDefensiveUtilityOfGameState(self,gameState):
      checkPoints= self.getBoardCheckPoints()
      enemy_locations =  [self.opponents_position_distribution[i].argMax() for i in range(len(self.opponents_position_distribution))]
      spottedOpponentOnOurHalf = False
      midPoint = int(gameState.data.layout.width/2)
    #   print('Midpoint', midPoint, flush=True)
      bestCheckPoint = 0
      for i in range(len(self.opponents_position_distribution)):
          enemy_location = enemy_locations[i]
          weight_of_enemy = self.opponents_position_distribution[i][tuple(enemy_location)]
        #   print('Opponent x coordinate ',list(enemy_location)[0],flush=True)
        #   print('weight of enemy',weight_of_enemy, flush=True)

          enemyPassedOnOurSide = enemy_location[0]>=midPoint
          if self.isOnRedTeam:
              enemyPassedOnOurSide = enemy_location[0]<midPoint

          if weight_of_enemy>0.0 and enemyPassedOnOurSide:
              spottedOpponentOnOurHalf = True
              bestCheckPoint = enemy_location
            #   print('Spotted opponent at position', enemy_location,flush=True)
      
      if spottedOpponentOnOurHalf==False:
        #   print('Our half is safe',flush=True)
          bestUtility=-np.inf
          for checkPoint in checkPoints:
              utility = self.checkPointUtility(gameState,checkPoint, enemy_locations)
              if utility>bestUtility:
                  bestUtility = utility
                  bestCheckPoint = checkPoint
        #   print('best utility', bestUtility,flush=True)
      
      # take distance to the best checkpint and dependent on that we weight the state

    #   print('My goal is at',bestCheckPoint,flush=True )
      ourPosition = gameState.getAgentPosition(self.index)
    #   print('I am at position',ourPosition, flush=True)
      utilityScore = -self.getMazeDistance(tuple(bestCheckPoint),tuple(ourPosition))

      return utilityScore
  
  def checkPointUtility(self,gameState, checkPoint, enemy_locations):
    
    minEnemyDistance=np.inf
    ourPosition = gameState.getAgentPosition(self.index)
    for enemy in enemy_locations:
        # print('checkpoint',tuple(checkPoint),flush=True)
        # print('Enemy', enemy, flush=True)
        distance = self.getMazeDistance(tuple(checkPoint), enemy)
        if distance<minEnemyDistance:
            minEnemyDistance = distance
    
    foodMatrix = self.getFoodYouAreDefending(self.gameState)
    minFoodDistance = np.inf
    for i in range(foodMatrix.width):
        for j in range(foodMatrix.height):
            if foodMatrix[i][j] == True:
                distance = self.getMazeDistance((i,j),tuple(checkPoint))
                if distance<minFoodDistance:
                    minFoodDistance = distance

    checkPointUtility = -(minFoodDistance + 2*minEnemyDistance)
    return checkPointUtility

  def getBoardCheckPoints(self):
      return self.halfPoints
  
  def getOffensiveUtilityOfGameState(self, gameState):
        
    gameScore = np.abs(gameState.getScore())     # Gets the current score of the game
    ourPosition = gameState.getAgentPosition(self.index)


    winScore = 0
    if gameState.isOver():
        if gameScore > 0:
            winScore = 1
        else:
            winScore = -1

    closestFoodDistance = self.getDistanceToEnemyFood(gameState)
    distanceToOurHalf = self.getMinDistanceToOurHalf(gameState)

    closestOpponent, enemyLocation = self.getDistanceToEnemyWhenOnTheirHalf(gameState)
    #-15*log (x/20)/log(20)
    distanceToClosestEnemy = self.getMazeDistance(tuple(enemyLocation),tuple(ourPosition))
    distanceToPill = self.closestPillDistance(gameState)


    maxDistance = 40
    food_Threshold = 1000
    # food_Threshold * np.log(np.abs(closestOpponent)/maxDistance) / np.log(maxDistance)
    # Here have some fancy utility function
    scoreUtility = 1000*gameScore
    winningUtility = 10000* winScore
    foodDistanceUtility =  10*closestFoodDistance
    # Todo make function here
    if self.scared:
        foodDepositUtility= 1000*self.foodWithin*distanceToOurHalf        # If we are scared we cannot eat so just run home and leave food
    else:
        foodDepositUtility= 10*self.foodWithin*distanceToOurHalf
    distanceToPillUtility = 10*distanceToPill
    enemyAvoidanceUtility = 5000

    if np.abs(distanceToClosestEnemy)<5:
        enemyAvoidanceUtility = food_Threshold * closestOpponent
        distanceToPillUtility = -10000*distanceToPillUtility
    #print('self.foodWithin',self.foodWithin, flush=True )
    #print('distanceToOurHalf',distanceToOurHalf, flush=True )


    #print('scoreUtility',scoreUtility, flush=True )
    #print('winningUtiliy',winningUtility, flush=True )
    #print('foodDistanceUtility',foodDistanceUtility, flush=True )
    #print('foodDepositUtility', foodDepositUtility, flush=True)
    #print('utility from avoiding enemy',enemyAvoidanceUtility, flush=True )

    # closestOpponent = 1 ->  10**4 * 1 = 10000
    # closestOpponent = 2 ->  10**3 * 2 = 2000

    #print('Food Deposit util', foodDepositUtility,flush=True)

    utilityScore = scoreUtility + winningUtility + foodDistanceUtility + foodDepositUtility + enemyAvoidanceUtility
    
    return utilityScore
  
  def closestPillDistance(self,gameState):
    capsuleList = self.getCapsules(gameState)
    minDistance = 0
    ourPosition = gameState.getAgentPosition(self.index)
    if len(capsuleList)>0:
        minDistance=np.inf
        for captule_positions in capsuleList:
            distance = self.getMazeDistance(captule_positions,ourPosition)
            if minDistance < distance:
                minDistance = distance

    return minDistance


  def getDistanceToEnemyWhenOnTheirHalf(self,gameState):
    if self.turn<1:
        return 1, list([1,2])
    opponent_position = [self.opponents_position_distribution[i].argMax() for i in range(len(self.opponents_position_distribution))]
    # print('opponent positions', opponent_position,flush=True)
    minDistance = 100000
    foodMatrix = self.getFood(gameState)
    x_dimension = foodMatrix.width
    midPoint = int(x_dimension/2)
    ourPosition = gameState.getAgentPosition(self.index)
    our_x = list(ourPosition)[0]
    areInDanger=False
    closestEnemyPosition=0
    enemy_counter=0
    for enemy in opponent_position:
        location = list(enemy)
        enemy_x = location[0]
        distance = self.getMazeDistance(enemy,ourPosition)
        enemyOnHisHalf = enemy_x < midPoint
        if self.isOnRedTeam:
            enemyOnHisHalf = enemy_x >= midPoint
        if distance < minDistance and enemyOnHisHalf:
            areInDanger=True
            opponent = gameState.data.agentStates[self.opponent_indeces[enemy_counter]]
            closestEnemyPosition=enemy
            minDistance = distance
        enemy_counter+=1

    if areInDanger:
        if opponent.scaredTimer>0:
            # print('I ate the pill!',flush=True)
            return -minDistance, list(closestEnemyPosition)
        else:
            return minDistance, list(closestEnemyPosition)
    else:
        return 1, list([1,2])

  def getNumberOfWallsAround(self,i,j,gameState):
    counter = 0
    if gameState.hasWall(i+1, j):
        counter+=1
    if gameState.hasWall(i, j+1):
        counter+=1
    if gameState.hasWall(i-1, j):
        counter+=1
    if gameState.hasWall(i, j-1):
        counter+=1
    return counter



  def populateHalfPoints(self,gameState):
    self.halfPoints=[]
    foodMatrix = self.getFood(gameState)

    x_dimension = foodMatrix.width
#    print('x dimension',x_dimension, flush=True)
#    print('my half point', int(x_dimension/2), flush=True )
    for j in range(foodMatrix.height): # vertical scan along the mid point of the terrain
        if self.isOnRedTeam:
            possibleOurSidePosition = [int(x_dimension/2-1), j] # we might need to change it if we are the other team
            enemySidePosition = [int(x_dimension/2), j] # one to the right
        else:
            possibleOurSidePosition = [int(x_dimension/2), j] # we might need to change it if we are the other team
            enemySidePosition = [int(x_dimension/2-1), j] # one to the left
        if not gameState.hasWall(enemySidePosition[0],enemySidePosition[1]) and not gameState.hasWall(possibleOurSidePosition[0],possibleOurSidePosition[1]):
            self.halfPoints.append(possibleOurSidePosition)


  def getMinDistanceToOurHalf(self,gameState):
#    print('We have this many halfPoints', self.halfPoints,flush=True)
    agentPosition = gameState.getAgentPosition(self.index)
    closestDistance = np.inf
    for position in self.halfPoints:
        distance = self.getMazeDistance(tuple(position),tuple(agentPosition))
        if distance < closestDistance:
            closestDistance=distance

    return -closestDistance

  def getDistanceToEnemyFood(self, gameState):
    agentPosition = gameState.getAgentPosition(self.index)
    foodMatrix = self.getFood(self.gameState)
    closestFoodLocationDistance = np.inf
    closest_x=0
    closest_y=0
    for i in range(foodMatrix.width):
        for j in range(foodMatrix.height):
            if foodMatrix[i][j] == True:
                numberOfWalls = self.getNumberOfWallsAround(i,j,gameState)
                coefficient = numberOfWalls
                if numberOfWalls<2: # not corner or corridor
                    coefficient = 1
                distance = coefficient*self.getMazeDistance((i,j),tuple(agentPosition))
                if distance<closestFoodLocationDistance:
                    closest_x=i
                    closest_y=j
                    closestFoodLocationDistance = distance
    return -closestFoodLocationDistance

