# Duels, set 1

Tasks measure a bot on its own. Duels measure it against the other bots on the leaderboard. Every pair of scored bots plays two kinds of duel, and the results are combined into one rating.

The reference implementation is [`src/boost_arena/duels.py`](../src/boost_arena/duels.py) and [`src/boost_arena/rating.py`](../src/boost_arena/rating.py).

## The two duels

| Duel | Setup | A point for |
|---|---|---|
| Penalty duel | The penalty shoot-out situation, with one bot attacking and the other in goal. 10 seconds. | The attacker if it scores, otherwise the keeper |
| Kickoff duel | Both bots start from a kickoff. 30 seconds. | Whoever scores first; no goal is a draw |

Each kind is played 200 times with bot A as blue and 200 times with bot B as blue, so both bots attack and defend the same situations. A pairing is therefore 800 episodes.

## The rating

Every point won and lost, across all duels and all opponents, goes into a Bradley-Terry fit. Each bot gets a strength such that the chance of A taking a point from B is A's strength divided by the two strengths together. The strengths are shown as a rating around 1000, with 400 points of difference meaning about ten points in eleven.

A small prior keeps a bot that has never lost a point from getting an infinite rating. A bot's rating changes when new bots arrive, because its opponents change; that is the nature of head-to-head ratings.

## When duels are played

- A bot duels only after it has an official task score.
- New pairs are played after each scoring run and once a week, up to 15 pairs per run. With many entrants a new bot may take a few runs to meet everyone.
- An updated bot plays everyone again after it has been scored again.
- The scoring service downloads a bot's sealed model from the address in its manifest for each duel run, so an entrant who removes their sealed file stops appearing in new duels. Their existing results stay.

## Replays

The first two episodes of each duel kind and direction are recorded, and can be watched from the pairing grid on the leaderboard.

## Trying it yourself

```bash
boost-arena duel my_bot.onnx other_bot.onnx
```

This plays 100 episodes per direction of each kind; the official number is 200.

## Versions

This is duel set 1. Changing a duel, a time limit or the number of episodes makes a new set, and ratings from different sets are not compared.
