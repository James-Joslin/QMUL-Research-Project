"""
This is a very simple form of twenty one. Ace only counts as value 1 not 1 or
11 for simplicity. This means that there is no such thing as a natural or two
card 21. This is a good example of showing how it can provide a good solution
to even luck based games.
"""

import numpy

from .abstract_game import AbstractGame


class Game(AbstractGame):
    """
    Game wrapper.
    """

    def __init__(self, seed=None):
        self.env = TwentyOne(seed)

    def step(self, action):
        """
        Apply action to the game.

        Args:
            action : action of the action_space to take.

        Returns:
            The new observation, the reward and a boolean if the game has ended.
        """
        observation, reward, done = self.env.step(action)
        return observation, reward * 10, done

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

    def human_to_action(self):
        """
        For multiplayer games, ask the user for a legal action
        and return the corresponding action number.

        Returns:
            An integer from the action space.
        """
        choice = input(f"Enter the action (0) Hit, or (1) Stand for the player {self.to_play()}: ")
        while choice not in [str(action) for action in self.legal_actions()]:
            choice = input("Enter either (0) Hit or (1) Stand : ")
        return int(choice)

    def action_to_string(self, action_number):
        """
        Convert an action number to a string representing the action.

        Args:
            action_number: an integer from the action space.

        Returns:
            String representing the action.
        """
        actions = {
            0: "Hit",
            1: "Stand",
        }
        return f"{action_number}. {actions[action_number]}"


class TwentyOne:
    def __init__(self, seed):
        self.random = numpy.random.RandomState(seed)

        self.player_hand = self.deal_card_value()
        self.dealer_hand = self.deal_card_value()

        self.player = 1

    def to_play(self):
        return 0 if self.player == 1 else 1

    def reset(self):
        self.player_hand = self.deal_card_value()
        self.dealer_hand = self.deal_card_value()
        self.player = 1
        return self.get_observation()

    """
    Action: 0 = Hit
    Action: 1 = Stand
    """

    def step(self, action):

        if action == 0:
            self.player_hand += self.deal_card_value()

        done = self.is_busted() or action == 1 or self.player_hand == 21

        if done:
            self.dealer_plays()

        return self.get_observation(), self.get_reward(done), done

    def get_observation(self):
        return [
            numpy.full((3, 3), self.player_hand, dtype="float32"),
            numpy.full((3, 3), self.dealer_hand, dtype="float32"),
            numpy.full((3, 3), 0),
        ]

    def legal_actions(self):
        # 0 = hit
        # 1 = stand
        return [0, 1]

    def get_reward(self, done):
        if not done:
            return 0
        if self.player_hand <= 21 and self.dealer_hand < self.player_hand:
            return 1
        if self.player_hand <= 21 and self.dealer_hand > 21:
            return 1
        if self.player_hand > 21:
            return -1
        if self.player_hand == self.dealer_hand:
            return 0
        return -1

    def deal_card_value(self):
        card = self.random.randint(1, 13)
        if card >= 10:
            value = 10
        else:
            value = card
        return value

    def dealer_plays(self):
        if self.player_hand > 21:
            return
        while self.dealer_hand <= 16:
            self.dealer_hand += self.deal_card_value()

    def is_busted(self):
        if self.player_hand > 21:
            return True

    def render(self):
        print("Dealer hand: " + str(self.dealer_hand))
        print("Player hand: " + str(self.player_hand))
