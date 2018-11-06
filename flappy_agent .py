from ple.games.flappybird import FlappyBird
from ple import PLE
import random

policyFile = ""
resultsFile = ""


class FlappyAgent:
    def __init__(self):

        return

    def reward_values(self):
        return {"positive": 1.0, "tick": 0.0, "loss": -5.0}

    def observe(self, s1, a, r, s2, end):
        return

    def training_policy(self, state):
        return random.randint(0, 1)

    def policy(self, state):
        return random.randint(0, 1)

    def getAllPossibleStates(self):
        allStates = []
        for i in range(15):
            for j in range(15):
                for k in range(15):
                    for l in range(15):
                        allStates.append((i, j, k, l))
        return allStates

    # statToTuple takes in the state from the PLE env and returns it as a
    # discretized with integer values ranged from 0 to 14

    def stateToTuple(self, state):
        ''' state =
            (
                Y position of bird,
                Distance from bird to next pipe,
                Vertical velocity of bird,
                height of pipe gap
            ) '''
        birdPosY = int(round(state["player_y"] * 14 / 512))
        pipeDist = int(round(state["next_pipe_dist_to_player"] * 14 / 288))
        pipeDist = 14 if pipeDist > 14 else pipeDist
        birdVelocity = int(round((state["player_vel"] + 8) * 14 / 18))
        birdVelocity = 0 if birdVelocity < 0 else birdVelocity
        pipeGapY = int(round(state["next_pipe_top_y"] * 14 / 512))
        return (birdPosY, pipeDist, birdVelocity, pipeGapY)

# This function takes in an existing policy and runs the game with it,
# writing down the score for each episode into a txt file and the training
# steps taken to get the corresponding policy


def test_policy(agent, frames, nb_test_episodes, policy):
    agent.q = policy
    reward_values = agent.reward_values()
    score = 0
    env = PLE(FlappyBird(), fps=30, display_screen=False, force_fps=True, rng=None,
              reward_values=reward_values)
    env.init()
    while nb_test_episodes > 0:
        state = env.game.getGameState()
        action = agent.policy(state)
        reward = env.act(env.getActionSet()[action])
        end = env.game_over()
        score += reward
        # reset the environment if the game is over
        if end:
            print("score for this episode: %d \t nr: %d   ------THIS IS FROM TEST------" %
                  (score, frames))
            env.reset_game()
            nb_test_episodes -= 1
            with open(resultsFile, "a") as f:
                f.write(f"{frames}:{score}\n")
            score = 0


# This function runs the game and trains.
# After every 10,000 training steps, the acquired policy is tested 10 times
# and the results written to resultsFile txt file
def train_and_test(nb_episodes, agent):
    reward_values = agent.reward_values()

    env = PLE(FlappyBird(), fps=30, display_screen=False, force_fps=True, rng=None,
              reward_values=reward_values)
    env.init()

    score = 0
    frames = 0
    test = True
    newGame = True
    while nb_episodes > 0 and frames < 1000000:
        if frames % 10000 == 0:
            test = True
        if test and newGame:
            test_policy(agent, frames, 10, agent.q)
            test = False
        newGame = False

        state = env.game.getGameState()
        action = agent.training_policy(state)
        s1 = agent.stateToTuple(state)
        reward = env.act(env.getActionSet()[action])
        s2 = agent.stateToTuple(env.game.getGameState())

        end = env.game_over()
        agent.observe(s1, action, reward, s2, end)

        score += reward

        frames += 1
        # reset the environment if the game is over
        if end:
            newGame = True
            print("score for this episode: %d \t nr: %d" %
                  (score, nb_episodes - 10000))
            env.reset_game()
            nb_episodes -= 1
            score = 0


class MonteCarlo(FlappyAgent):
    def __init__(self):
        # The q and pi dictionaries are all initialiez with all 50,625 different
        # possible states as keys, with the value (0,0)
        self.q = dict.fromkeys(self.getAllPossibleStates(), (0, 0))
        self.pi = dict.fromkeys(self.getAllPossibleStates(), (0, 0))
        # The episode is for storing every step taken in an episode when training
        self.episode = []
        # The keys in the returns dictionary are the (state + action) we have gotten values from
        # and the values as a list for each (state + action) visited
        self.returns = dict()
        self.discount = 1
        return

    # In MC we call this function after every episode finished when training.
    def learn_from_episode(self):
        NextReward = 0
        # We loop through each step in the reversed episode, appending the new value
        # for each state visited to the returns list, and then recalculating the average
        # value for the state in the q dictionary
        for G in reversed(self.episode):
            # value to append in returns:
            #   reward + reward * discount + reward of next step
            reward = self.discount * G[2] + NextReward
            stateAction = G[0] + (G[1],)
            if not stateAction in self.returns:
                self.returns[stateAction] = [reward]
            else:
                self.returns[stateAction].append(reward)

            newMean = sum(self.returns[stateAction]) / \
                len(self.returns[stateAction])
            if G[1] == 1:
                newActionValue = (self.q[G[0]][0], newMean)
            else:
                newActionValue = (newMean, self.q[G[0]][1])
            self.q[G[0]] = newActionValue
            NextReward = reward

            # Then we change the pi:
            #   if action = A* then value = 1 - epsilon + epsilon / 2
            #   else value = epsilon / 2, with ties broken arbitrarily
            AstarValue = 1 - 0.1 + 0.1 / 2
            nonAstarValue = 0.1 / 2

            if self.q[G[0]][0] > self.q[G[0]][1]:
                self.pi[G[0]] = (AstarValue, nonAstarValue)
            elif self.q[G[0]][0] < self.q[G[0]][1] or random.randint(0, 1) == 1:
                self.pi[G[0]] = (nonAstarValue, AstarValue)
            else:
                self.pi[G[0]] = (AstarValue, nonAstarValue)
        return

    # After every training step we append the results to the episode list.
    # If the episode is finished we learn from it and reset it.
    def observe(self, s1, a, r, s2, end):
        self.episode.append((s1, a, r, s2))
        if end:
            self.learn_from_episode()
            self.episode = []
        return

    # When training there is a 10% chance we explore, else we
    # use the policy from pi with ties broken arbitrarily
    def training_policy(self, state):
        if random.random() < 0.1:
            return random.randint(0, 1)

        actionTuple = self.pi[self.stateToTuple(state)]
        if actionTuple[1] > actionTuple[0]:
            return 1
        elif actionTuple[1] < actionTuple[0]:
            return 0
        return random.randint(0, 1)

    # Returns 0 if q[state][0] is greater than q[state][1]
    def policy(self, state):
        actionTuple = self.q[self.stateToTuple(state)]
        if actionTuple[0] > actionTuple[1]:
            return 0
        return 1


class QLearning(FlappyAgent):
    def __init__(self):
        # We initialize the q dictionary with all possible states as keys
        # and their value as 0
        self.q = dict.fromkeys(self.getAllPossibleStates(), (0, 0))
        self.discount = 1
        return

    # In Q-Learning we learn after every step taken in an episode
    def learn_from_step(self, s1, a, r, s2):
        # We set the new value of for an action in the state as:
        #   old value + 0.1 * (reward + discount * (value of highest action in s2) - old value)
        nextMax = max([self.q[s2][0], self.q[s2][1]])
        newVal = self.q[s1][a] + 0.1 * \
            (r + self.discount * nextMax - self.q[s1][a])

        if a == 1:
            newActionValue = (self.q[s1][0], newVal)
        else:
            newActionValue = (newVal, self.q[s1][1])
        self.q[s1] = newActionValue
        return

    # Observed is called after each training step
    def observe(self, s1, a, r, s2, end):
        self.learn_from_step(s1, a, r, s2)
        return

    # The training policy uses the optimal policy aquired but with
    # a 10% chance of exploring
    def training_policy(self, state):
        actionTuple = self.q[self.stateToTuple(state)]
        if random.random() < 0.1:
            return random.randint(0, 1)
        if actionTuple[1] > actionTuple[0]:
            return 1
        elif actionTuple[1] < actionTuple[0]:
            return 0
        return random.randint(0, 1)

    def policy(self, state):
        actionTuple = self.q[self.stateToTuple(state)]
        if actionTuple[1] > actionTuple[0]:
            return 1
        elif actionTuple[1] < actionTuple[0]:
            return 0
        return random.randint(0, 1)


agent = MonteCarlo()
policyFile = "MC_Policy.txt"
resultsFile = "MC_Results.txt"

train_and_test(10000, agent)

# Use this instead if testing an existing policy:
# policy = eval(open(policyFile, 'r').read())
# test_policy(agent, 0, 10, policy)


with open(policyFile, "w") as f:
    f.write(str(agent.q))
