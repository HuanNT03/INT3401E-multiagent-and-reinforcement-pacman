# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        # Heursitics components
        score = 0

        newFoodPositions = newFood.asList() 
        newFoodDistances = [manhattanDistance(newPos, foodPos) for foodPos in newFoodPositions]

        newGhostPositions = [ghost.getPosition() for ghost in newGhostStates]
        newGhostDistances = [manhattanDistance(newPos, ghostPos) for ghostPos in newGhostPositions]
        closestGhost = min(newGhostDistances)
        # ghostPenalty = sum(1 / distance if newScaredTimes[newGhostDistances.index(distance)] == 0 and distance > 0 else distance for distance in newGhostDistances)
        
        # If action is stop will reduce the score
        if action == 'Stop':
            score -= 50

        if len(newFoodDistances) == 0:
            return successorGameState.getScore() + score + closestGhost

        closestFood = min(newFoodDistances) 
        
        score += (closestGhost / (closestFood * 10))

        return successorGameState.getScore() + score

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        def minimax(gameState, depth, agentID):
            if depth == self.depth or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            if agentID == 0:
                if len(gameState.getLegalActions()) == 0:
                    return self.evaluationFunction(gameState)
                v = float('-inf')
                ac = Directions.STOP
                for a in gameState.getLegalActions():
                    s = minimax(gameState.generateSuccessor(0, a), depth, 1)
                    if s > v:
                        v = s
                        ac = a
                if depth == 0:
                    return ac
                else:
                    return v
            else:
                if len(gameState.getLegalActions()) == 0:
                    return self.evaluationFunction(gameState)
                v = float('inf')
                nextAgent = agentID + 1
                if nextAgent == gameState.getNumAgents():
                    nextAgent = 0
                for a in gameState.getLegalActions(agentID):
                    if nextAgent == 0:
                        s = minimax(gameState.generateSuccessor(agentID, a), depth + 1, 0)
                    else:
                        s = minimax(gameState.generateSuccessor(agentID, a), depth, nextAgent)
                    v = min(s, v)
                return v

        return minimax(gameState, 0, 0)

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        # Format of result = [action, score]
        # Initial state: index = 0, depth = 0, alpha = -infinity, beta = +infinity
        result = self.getBestActionAndScore(gameState, 0, 0, float("-inf"), float("inf"))

        # Return the action from result
        return result[0]

    def getBestActionAndScore(self, gameState, index, depth, alpha, beta):
        """
        Returns value as pair of [action, score] based on the different cases:
        1. Terminal state
        2. Max-agent
        3. Min-agent
        """
        # Terminal states:
        if len(gameState.getLegalActions(index)) == 0 or depth == self.depth:
            return "", gameState.getScore()

        # Max-agent: Pacman has index = 0
        if index == 0:
            return self.max_value(gameState, index, depth, alpha, beta)

        # Min-agent: Ghost has index > 0
        else:
            return self.min_value(gameState, index, depth, alpha, beta)

    def max_value(self, gameState, index, depth, alpha, beta):
        """
        Returns the max utility action-score for max-agent with alpha-beta pruning
        """
        legalMoves = gameState.getLegalActions(index)
        max_value = float("-inf")
        max_action = ""

        for action in legalMoves:
            successor = gameState.generateSuccessor(index, action)
            successor_index = index + 1
            successor_depth = depth

            # Update the successor agent's index and depth if it's pacman
            if successor_index == gameState.getNumAgents():
                successor_index = 0
                successor_depth += 1

            # Calculate the action-score for the current successor
            current_action, current_value \
                = self.getBestActionAndScore(successor, successor_index, successor_depth, alpha, beta)

            # Update max_value and max_action for maximizer agent
            if current_value > max_value:
                max_value = current_value
                max_action = action

            # Update alpha value for current maximizer
            alpha = max(alpha, max_value)

            # Pruning: Returns max_value because next possible max_value(s) of maximizer
            # can get worse for beta value of minimizer when coming back up
            if max_value > beta:
                return max_action, max_value

        return max_action, max_value

    def min_value(self, gameState, index, depth, alpha, beta):
        """
        Returns the min utility action-score for min-agent with alpha-beta pruning
        """
        legalMoves = gameState.getLegalActions(index)
        min_value = float("inf")
        min_action = ""

        for action in legalMoves:
            successor = gameState.generateSuccessor(index, action)
            successor_index = index + 1
            successor_depth = depth

            # Update the successor agent's index and depth if it's pacman
            if successor_index == gameState.getNumAgents():
                successor_index = 0
                successor_depth += 1

            # Calculate the action-score for the current successor
            current_action, current_value \
                = self.getBestActionAndScore(successor, successor_index, successor_depth, alpha, beta)

            # Update min_value and min_action for minimizer agent
            if current_value < min_value:
                min_value = current_value
                min_action = action

            # Update beta value for current minimizer
            beta = min(beta, min_value)

            # Pruning: Returns min_value because next possible min_value(s) of minimizer
            # can get worse for alpha value of maximizer when coming back up
            if min_value < alpha:
                return min_action, min_value

        return min_action, min_value

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        numAgents = gameState.getNumAgents()

        def expectimax(gameState, depth, agentIndex):
            if gameState.isWin() or gameState.isLose() or (depth == self.depth and agentIndex == 0):
                return None, self.evaluationFunction(gameState)

            best_move = None
            best_val = float("-inf") if agentIndex == 0 else 0

            nextDepth = depth + 1 if agentIndex == (numAgents - 1) else depth

            actions = gameState.getLegalActions(agentIndex)

            for move in actions:
                nextGameState = gameState.generateSuccessor(agentIndex, move)

                _, next_val = expectimax(nextGameState, nextDepth, (agentIndex + 1) % numAgents)
                if agentIndex == 0:
                    if best_val < next_val:
                        best_move, best_val = move, next_val
                else:
                    best_val += (1.0 / len(actions)) * next_val
            return best_move, best_val

        return expectimax(gameState, 0, 0)[0]

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    # Extract information
    pos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood()
    capsules = currentGameState.getCapsules()
    ghostStates = currentGameState.getGhostStates()
    
    score = currentGameState.getScore()  # Base score
    
    # Food: Reward being closer to food
    foodDist = [manhattanDistance(pos, foodPos) for foodPos in food.asList()]
    score += sum(1.0 / d if d > 0 else 10 for d in foodDist)  # Avoid division by zero

    # Ghosts: Penalize proximity to active ghosts, reward scared ghosts
    for ghost in ghostStates:
        ghostDist = manhattanDistance(pos, ghost.getPosition())
        if ghost.scaredTimer > 0:  # Scared ghost
            score += 200 / (ghostDist + 1)
        else:  # Active ghost
            score -= 200 / (ghostDist + 1)

    # Capsules: Encourage capsule collection
    capsuleDist = [manhattanDistance(pos, cap) for cap in capsules]
    score += sum(100 / (d + 1) for d in capsuleDist)
    
    return score

# Abbreviation
better = betterEvaluationFunction
