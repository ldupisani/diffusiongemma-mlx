"""Game theory — 2x2 zero-sum value, pure Nash equilibria, expected payoff, dominant
strategy. Coded, 0 calls."""
from reasoning.bind import bind_game
from reasoning.compiled import dispatch

META = {
    "id": "game_theory", "query_type": "game_theory",
    "description": "2x2 game analysis: value of a zero-sum game, pure-strategy Nash equilibria "
                   "of a bimatrix game, expected payoff under mixed strategies, dominant strategy",
    "example_queries": [
        "What is the value of the zero-sum game [[1,-1],[-1,1]]?",
        "Find the pure Nash equilibria of row payoffs [[3,0],[5,1]] and column payoffs [[3,5],[0,1]].",
        "What is the expected payoff for [[3,1],[0,2]] with p=0.5 and q=0.5?",
        "What is the dominant strategy for [[3,1],[0,0]]?",
    ],
    "keywords": ["game theory", "nash equilibrium", "zero-sum", "payoff", "dominant strategy"],
    # A precise DOMAIN orchestrator (tier 2), not bind-gated: its conservative precondition
    # gates candidacy, and its keyword-rich examples win the lexical match over a generic
    # number-list orchestrator that would otherwise grab the matrix entries.
    "model_free": True,
}


def precondition(query):
    return bind_game(query)[0]


def run(query):
    return dispatch(bind_game(query), parse_message="could not parse the game-theory query")
