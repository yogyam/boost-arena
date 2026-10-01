"""A rating from duel results: the Bradley-Terry model.

Each bot gets a strength. The chance that bot A takes a point from bot B is
strength_A / (strength_A + strength_B). The strengths are fitted to all the duel points
by the usual iterative method, then shown on a scale where the average bot is 1000 and
a 400-point lead means winning about ten points in eleven.
"""

import math


def fit_strengths(points: dict, iterations: int = 200, prior: float = 1.0) -> dict:
    """`points[(a, b)]` is the number of points a took from b. Returns strength per bot.

    A small prior, as if every pair had exchanged `prior` points each way, keeps a bot
    that has never lost from getting an infinite strength.
    """
    bots = sorted({bot for pair in points for bot in pair})
    if not bots:
        return {}
    strength = {bot: 1.0 for bot in bots}
    pairs = {}
    for (a, b), won in points.items():
        pairs[(a, b)] = pairs.get((a, b), 0.0) + won
    for a in bots:
        for b in bots:
            if a != b:
                pairs[(a, b)] = pairs.get((a, b), 0.0) + prior

    for _ in range(iterations):
        new = {}
        for a in bots:
            wins = sum(won for (x, _), won in pairs.items() if x == a)
            denominator = 0.0
            for b in bots:
                if b == a:
                    continue
                games = pairs.get((a, b), 0.0) + pairs.get((b, a), 0.0)
                denominator += games / (strength[a] + strength[b])
            new[a] = wins / denominator if denominator else 1.0
        mean = math.exp(sum(math.log(s) for s in new.values()) / len(new))
        strength = {bot: s / mean for bot, s in new.items()}
    return strength


def ratings(points: dict) -> dict:
    """Bradley-Terry strengths as ratings around 1000, 400 points per factor of ten."""
    return {bot: round(1000 + 400 * math.log10(s), 1) for bot, s in fit_strengths(points).items()}


def expected_share(rating_a: float, rating_b: float) -> float:
    """The share of points A is expected to take from B."""
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
