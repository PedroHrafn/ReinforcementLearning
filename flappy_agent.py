from ple.games.flappybird import FlappyBird
from ple import PLE
import random


class FlappyAgent:
    def __init__(self):

        # TODO: you may need to do some initialization for your agent here
        self.q = dict.fromkeys(self.getAllPossibleStates(), (0, 0))
        self.pi = dict.fromkeys(self.getAllPossibleStates(), (0, 0))
        self.episode = []
        self.returns = dict()
        self.discount = 1
        return

    def getAllPossibleStates(self):
        allStates = []
        for i in range(15):
            for j in range(15):
                for k in range(15):
                    for l in range(15):
                        allStates.append((i, j, k, l))
        return allStates

    def reward_values(self):
        """ returns the reward values used for training

            Note: These are only the rewards used for training.
            The rewards used for evaluating the agent will always be
            1 for passing through each pipe and 0 for all other state
            transitions.
        """
        return {"positive": 1.0, "tick": 0.0, "loss": -5.0}

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
        """ this function is called during training on each step of the game where
            the state transition is going from state s1 with action a to state s2 and
            yields the reward r. If s2 is a terminal state, end==True, otherwise end==False.

            Unless a terminal state was reached, two subsequent calls to observe will be for
            subsequent steps in the same episode. That is, s1 in the second call will be s2
            from the first call.
            """
        # TODO: learn from the observation

        self.episode.append((s1, a, r, s2))
        if end:
            self.learn_from_episode()
            self.episode = []

        return

    def stateToTuple(self, state):
        """ position of bird: y_axis    [0,512]
            distanace to pipe: x_axis   [0,288]
            vertical velocity of the bird[-8,,10] per frame
            y-coords of pipe-gap        [0,512] """

        birdPosY = int(round(state["player_y"] * 14 / 512))
        pipeDist = int(round(state["next_pipe_dist_to_player"] * 14 / 288))
        pipeDist = 14 if pipeDist > 14 else pipeDist
        birdVelocity = int(round((state["player_vel"] + 8) * 14 / 18))
        birdVelocity = 0 if birdVelocity < 0 else birdVelocity
        pipeGapY = int(round(state["next_pipe_top_y"] * 14 / 512))

        return (birdPosY, pipeDist, birdVelocity, pipeGapY)

    def training_policy(self, state):
        """ Returns the index of the action that should be done in state while training the agent.
            Possible actions in Flappy Bird are 0 (flap the wing) or 1 (do nothing).

            training_policy is called once per frame in the game while training
        """
        if random.random() < 0.1:
            return random.randint(0, 1)

        actionTuple = self.pi[self.stateToTuple(state)]
        if actionTuple[1] > actionTuple[0]:
            return 1
        elif actionTuple[1] < actionTuple[0]:
            return 0
        return random.randint(0, 1)

    def policy(self, state):
        """ Returns the index of the action that should be done in state when training is completed.
            Possible actions in Flappy Bird are 0 (flap the wing) or 1 (do nothing).

            policy is called once per frame in the game (30 times per second in real-time)
            and needs to be sufficiently fast to not slow down the game.
        """
        #print("state: %s" % state)
        # TODO: change this to to policy the agent has learned
        # At the moment we just return an action uniformly at random.
        """ if (state["next_pipe_bottom_y"] > state["player_y"] + 40):
            return 1 """
        actionTuple = self.q[self.stateToTuple(state)]
        if actionTuple[1] >= actionTuple[0]:
            return 1
        return 0


def run_game(nb_episodes, agent):
    """ Runs nb_episodes episodes of the game with agent picking the moves.
        An episode of FlappyBird ends with the bird crashing into a pipe or going off screen.
    """

    """ reward_values = {"positive": 1.0, "negative": 0.0,
                     "tick": 0.0, "loss": 0.0, "win": 0.0} """
    # TODO: when training use the following instead:
    reward_values = agent.reward_values()

    env = PLE(FlappyBird(), fps=30, display_screen=True, force_fps=True, rng=None,
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
        action = agent.training_policy(state)
        # action = agent.policy(env.game.getGameState())
        # step the environment
        s1 = agent.stateToTuple(state)
        reward = env.act(env.getActionSet()[action])
        #print("reward=%d" % reward)
        s2 = agent.stateToTuple(env.game.getGameState())
        # TODO: for training let the agent observe the current state transition
        end = env.game_over()
        agent.observe(s1, action, reward, s2, end)

        score += reward

        frames += 1
        # reset the environment if the game is over
        if end:
            print("score for this episode: %d \t nr: %d" %
                  (score, nb_episodes - 10000))
            env.reset_game()
            nb_episodes -= 1
            with open("MC_Results.txt", "a") as f:
                f.write(f"{frames}:{score}\n")
            score = 0


agent = FlappyAgent()
with open("MC_Policy.txt", "w") as f:
    f.write("havent done anything")
run_game(10000, agent)

with open("MC_Policy.txt", "w") as f:
    f.write(str(agent.q))


""" plot2D = [[0] * 15 for i in range(15)]
for x in plot2D:
    print(x) """
