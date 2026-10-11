"""Optional game from MuZero-General, Werner Duvaud (MIT)."""

import numpy

from .abstract_game import AbstractGame


class Game(AbstractGame):
    """
    Game wrapper.
    """

    def __init__(self, seed=None):
        self.env = GridEnv()

    def step(self, action):
        """
        Apply action to the game.

        Args:
            action : action of the action_space to take.

        Returns:
            The new observation, the reward and a boolean if the game has ended.
        """
        observation, reward, done = self.env.step(action)
        return [[observation]], reward * 10, done

    def legal_actions(self):
        """
        Should return the legal actions at each turn, if it is not available, it can return
        the whole action space. At each turn, the game have to be able to handle one of returned actions.

        For complex game where calculating legal moves is too long, the idea is to define the legal actions
        equal to the action space but to return a negative reward if the action is illegal.

        Returns:
            An array of integers, subset of the action space.
        """
        return list(range(2))

    def reset(self):
        """
        Reset the game for a new game.

        Returns:
            Initial observation of the game.
        """
        return [[self.env.reset()]]

    def render(self):
        """
        Display the game observation.
        """
        self.env.render()

    def action_to_string(self, action_number):
        """
        Convert an action number to a string representing the action.

        Args:
            action_number: an integer from the action space.

        Returns:
            String representing the action.
        """
        actions = {
            0: "Down",
            1: "Right",
        }
        return f"{action_number}. {actions[action_number]}"


class GridEnv:
    def __init__(self, size=3):
        self.size = size
        self.position = [0, 0]

    def legal_actions(self):
        legal_actions = list(range(2))
        if self.position[0] == (self.size - 1):
            legal_actions.remove(0)
        if self.position[1] == (self.size - 1):
            legal_actions.remove(1)
        return legal_actions

    def step(self, action):
        if action not in self.legal_actions():
            pass
        elif action == 0:
            self.position[0] += 1
        elif action == 1:
            self.position[1] += 1

        reward = 1 if self.position == [self.size - 1] * 2 else 0
        return self.get_observation(), reward, bool(reward)

    def reset(self):
        self.position = [0, 0]
        return self.get_observation()

    def render(self):
        im = numpy.full((self.size, self.size), "-")
        im[self.size - 1, self.size - 1] = "1"
        im[self.position[0], self.position[1]] = "x"
        print(im)

    def get_observation(self):
        observation = numpy.zeros((self.size, self.size))
        observation[self.position[0]][self.position[1]] = 1
        return observation.flatten()
