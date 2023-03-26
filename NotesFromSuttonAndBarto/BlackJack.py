import numpy as np
from dataclasses import dataclass
@dataclass
class BlackJackState:
    score: int
    usable: bool
    dealer_showing: object

    def int_vals(self):
        x = self.score - 12
        y=int(self.usable)
        z = 0 if self.dealer_showing == 'A' else int(self.dealer_showing) - 1
        return x,y,z
class BlackJack:
    vals = ['A', 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]

    def __init__(self):
        self.game_state = None
        self.aces = None
        self.player_non_ace_cards = None
        self.dealer_aces = None
        self.dealer_non_ace_cards = None
        self.dealer_hidden = None
        self.dealer_shown = None

    def reset(self):
        self.dealer_shown = np.random.choice(self.vals)
        self.dealer_hidden = np.random.choice(self.vals)

        self.dealer_non_ace_cards = []
        self.dealer_aces = 0

        for c in (self.dealer_shown, self.dealer_hidden):
            if c == 'A':
                self.dealer_aces += 1
            else:
                self.dealer_non_ace_cards.append(int(c))

        self.player_non_ace_cards = []
        self.aces = 0

        self.game_state = 1  # game not truncated

        while self.score < 12:
            self.hit()

        if self.score == 21:
            self.game_state = 0

        r = self._result if self.game_state == 0 else 0

        return self.state, r, not bool(self.game_state)

    def step(self, action):
        # action 1 is hit and 0 is stick
        assert action in (0, 1), "Action {} not allowed".format(action)
        if action == 1:
            self.hit()
            r = self._result if self.game_state == 0 else 0
            return self.state, r, not bool(self.game_state)
        else:
            self.stick()
            return self.state, self._result, not bool(self.game_state)

    @property
    def score(self):
        t = sum(self.player_non_ace_cards)
        a = self.aces
        if a:
            if 21 - t > 11:
                t += 11
                a -= 1
        if a:
            t += a
        return t

    @property
    def state(self):
        t = sum(self.player_non_ace_cards)
        usable = 21 - t > 11 and self.aces
        return BlackJackState(self.score, bool(usable), self.dealer_shown)

    @property
    def _dealer_score(self):
        t = sum(self.dealer_non_ace_cards)
        a = self.dealer_aces
        if a:
            if 21 - t > 11:
                t += 11
                a -= 1
        if a:
            t += a
        return t

    def hit(self):
        if self.game_state == 0:
            raise Exception('Game Ended')
        v = np.random.choice(self.vals)
        if v == 'A':
            self.aces += 1
        else:
            self.player_non_ace_cards.append(int(v))
        if self.score > 21:
            self.game_state = 0

    def _dealer_hit(self):
        v = np.random.choice(self.vals)
        if v == 'A':
            self.dealer_aces += 1
        else:
            self.dealer_non_ace_cards.append(int(v))

        if self._dealer_score > 21:
            self.game_state = 0
        return self.game_state, self._dealer_score

    def stick(self):
        if self.game_state == 0:
            raise Exception('Game Ended')
        while self._dealer_score < 17:
            self._dealer_hit()
        self.game_state = 0

    @property
    def _result(self):
        assert self.game_state == 0, "_result Called Before Game Ended"
        if self.score > 21:
            return -1
        elif self._dealer_score > 21:
            return 1
        elif self.score > self._dealer_score:
            return 1
        elif self.score < self._dealer_score:
            return -1
        else:
            return 0

    def check_natural(self):
        if self.score == 21:
            if self._dealer_score == 21:
                return 0
            else:
                return 1