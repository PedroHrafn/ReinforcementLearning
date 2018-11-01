from ple.games.flappybird import FlappyBird
from ple import PLE
import random


class FlappyAgent:
    def __init__(self):
        # TODO: you may need to do some initialization for your agent here
        return

    def reward_values(self):
        return {"positive": 1.0, "tick": 0.0, "loss": -5.0}

    def observe(self, s1, a, r, s2, end):
        # TODO: learn from the observation
        return

    def training_policy(self, state):
        # TODO: change this to to policy the agent is supposed to use while training
        # At the moment we just return an action uniformly at random.
        return random.randint(0, 1)

    def getAllPossibleStates(self):
        allStates = []
        for i in range(15):
            for j in range(15):
                for k in range(15):
                    for l in range(15):
                        allStates.append((i, j, k, l))
        return allStates

    def stateToTuple(self, state):
        birdPosY = int(round(state["player_y"] * 14 / 512))
        pipeDist = int(round(state["next_pipe_dist_to_player"] * 14 / 288))
        pipeDist = 14 if pipeDist > 14 else pipeDist
        birdVelocity = int(round((state["player_vel"] + 8) * 14 / 18))
        birdVelocity = 0 if birdVelocity < 0 else birdVelocity
        pipeGapY = int(round(state["next_pipe_top_y"] * 14 / 512))
        return (birdPosY, pipeDist, birdVelocity, pipeGapY)

    def policy(self, state):
        return random.randint(0, 1)


def run_game(nb_episodes, agent):
    # TODO: when training use the following instead:
    reward_values = agent.reward_values()

    env = PLE(FlappyBird(), fps=30, display_screen=True, force_fps=False, rng=None,
              reward_values=reward_values)
    # TODO: to speed up training change parameters of PLE as follows:
    # display_screen=False, force_fps=True
    env.init()

    score = 0
    frames = 0
    while nb_episodes > 0 and frames < 1000000:
        # pick an action
        # TODO: for training using agent.training_policy instead
        state = env.game.getGameState()
        # action = agent.training_policy(state)
        action = agent.policy(env.game.getGameState())
        # step the environment
        s1 = agent.stateToTuple(state)
        reward = env.act(env.getActionSet()[action])
        #print("reward=%d" % reward)
        s2 = agent.stateToTuple(env.game.getGameState())
        # TODO: for training let the agent observe the current state transition
        end = env.game_over()
        #agent.observe(s1, action, reward, s2, end)

        score += reward

        frames += 1
        # reset the environment if the game is over
        if end:
            print("score for this episode: %d \t nr: %d" %
                  (score, nb_episodes - 10000))
            env.reset_game()
            nb_episodes -= 1
            with open("QL_Results.txt", "a") as f:
                f.write(f"{frames}:{score}\n")
            score = 0


class MonteCarlo(FlappyAgent):
    def __init__(self):
        self.q = dict.fromkeys(self.getAllPossibleStates(), (0, 0))
        self.pi = dict.fromkeys(self.getAllPossibleStates(), (0, 0))
        self.episode = []
        self.returns = dict()
        self.discount = 1
        return

    def learn_from_episode(self):
        NextReward = 0
        for G in reversed(self.episode):
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

            AstarValue = 1 - 0.1 + 0.1 / 2
            nonAstarValue = 0.1 / 2

            if self.q[G[0]][0] > self.q[G[0]][1]:
                self.pi[G[0]] = (AstarValue, nonAstarValue)
            elif self.q[G[0]][0] < self.q[G[0]][1] or random.randint(0, 1) == 1:
                self.pi[G[0]] = (nonAstarValue, AstarValue)
            else:
                self.pi[G[0]] = (AstarValue, nonAstarValue)
            NextReward = reward
        return

    def observe(self, s1, a, r, s2, end):
        self.episode.append((s1, a, r, s2))
        if end:
            self.learn_from_episode()
            self.episode = []
        return

    def training_policy(self, state):
        if random.random() < 0.1:
            return random.randint(0, 1)

        actionTuple = self.pi[self.stateToTuple(state)]
        if actionTuple[1] > actionTuple[0]:
            return 1
        elif actionTuple[1] < actionTuple[0]:
            return 0
        return random.randint(0, 1)

    def policy(self, state):
        actionTuple = self.q[self.stateToTuple(state)]
        if actionTuple[1] >= actionTuple[0]:
            return 1
        return 0


class QLearning(FlappyAgent):
    def __init__(self):
        self.q = dict.fromkeys(self.getAllPossibleStates(), (0, 0))
        self.discount = 1
        return

    def learn_from_step(self, s1, a, r, s2):
        nextMax = max([self.q[s2][0], self.q[s2][1]])
        newVal = self.q[s1][a] + 0.1 * \
            (r + self.discount * nextMax - self.q[s1][a])

        if a == 1:
            newActionValue = (self.q[s1][0], newVal)
        else:
            newActionValue = (newVal, self.q[s1][1])
        self.q[s1] = newActionValue
        return

    def observe(self, s1, a, r, s2, end):
        self.learn_from_step(s1, a, r, s2)
        return

    def training_policy(self, state):
        actionTuple = self.q[self.stateToTuple(state)]
        if actionTuple[1] > actionTuple[0]:
            return 1
        return 0

    def policy(self, state):
        actionTuple = self.q[self.stateToTuple(state)]
        if actionTuple[1] > actionTuple[0]:
            return 1
        return 0


agent = QLearning()
agent.q = eval(open('QL_Policy.txt', 'r').read())

run_game(10000, agent)
with open("QL_Policy.txt", "w") as f:
    f.write(str(agent.q))
