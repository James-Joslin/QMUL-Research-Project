"""Optional game from MuZero-General, Werner Duvaud (MIT)."""

import numpy

from .abstract_game import AbstractGame

# This is a Game wrapper for open_spiel games. It allows you to run any game in the open_spiel library.

try:
    import pyspiel

except ImportError as error:
    raise ModuleNotFoundError("The spiel adapter requires open_spiel") from error

# The game you want to run. See https://github.com/deepmind/open_spiel/blob/master/docs/games.md for a list of games
game = pyspiel.load_game("tic_tac_toe")


class Game(AbstractGame):
    """
    Game wrapper.
    """

    def __init__(self, seed=None):
        self.env = Spiel()

    def step(self, action):
        """
        Apply action to the game.

        Args:
            action : action of the action_space to take.

        Returns:
            The new observation, the reward and a boolean if the game has ended.
        """
        observation, reward, done = self.env.step(action)
        return observation, reward * 20, done

    def to_play(self):
        """
        Return the current player.

        Returns:
            The current player, it should be an element of the players list in the config.
        """
        return self.env.to_play()

    def legal_actions(self):
        """
        Should return the legal actions at each turn, if it is not available, it can return
        the whole action space. At each turn, the game have to be able to handle one of returned actions.

        For complex game where calculating legal moves is too long, the idea is to define the legal actions
        equal to the action space but to return a negative reward if the action is illegal.

        Returns:
            An array of integers, subset of the action space.
        """
        return self.env.legal_actions()

    def reset(self):
        """
        Reset the game for a new game.

        Returns:
            Initial observation of the game.
        """
        return self.env.reset()

    def render(self):
        """
        Display the game observation.
        """
        self.env.render()

    def legal_actions_human(self):
        return self.env.human_legal_actions()

    def human_to_action(self):
        """
        For multiplayer games, ask the user for a legal action
        and return the corresponding action number.

        Returns:
            An integer from the action space.
        """
        while True:
            try:
                print("Legal Actions: ", self.legal_actions_human())
                choice = input("Enter your move: ")
                if choice in self.legal_actions_human():
                    break
            except ValueError:
                pass
            print("Wrong input, try again")

        return self.env.board.string_to_action(choice)

    def action_to_string(self, action_number):
        """
        Convert an action number to a string representing the action.

        Args:
            action_number: an integer from the action space.

        Returns:
            String representing the action.
        """
        row = action_number // 3 + 1
        col = action_number % 3 + 1
        return f"Play row {row}, column {col}"


class Spiel:
    def __init__(self):
        self.game = game
        self.board = self.game.new_initial_state()
        self.player = 1

    def to_play(self):
        return 0 if self.player == 1 else 1

    def reset(self):
        self.board = self.game.new_initial_state()
        self.player = 1
        return self.get_observation()

    def step(self, action):
        self.board = self.board.child(action)

        done = self.board.is_terminal()

        reward = 1 if self.have_winner() else 0

        observation = self.get_observation()

        self.player *= -1

        return observation, reward, done

    def get_observation(self):
        if self.player == 1:
            current_player = 1
        else:
            current_player = 0
        return numpy.array(self.board.observation_tensor(current_player)).reshape(
            self.game.observation_tensor_shape()
        )

    def legal_actions(self):
        return self.board.legal_actions()

    def have_winner(self):
        rewards = self.board.rewards()

        if self.player == 1:
            if rewards[0] == 1.0:
                return True

        elif self.player == -1:
            if rewards[1] == 1.0:
                return True

        return False

    def human_legal_actions(self):
        return [self.board.action_to_string(x) for x in self.board.legal_actions()]

    def render(self):
        print(self.board)
