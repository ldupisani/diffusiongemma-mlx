"""Mathematical statistics — correlation, least-squares regression, z-scores. Coded, 0 calls."""
from reasoning.bind import bind_statistics
from reasoning.compiled import dispatch

META = {
    "id": "statistics", "query_type": "statistics",
    "description": "bivariate / inferential statistics: Pearson correlation between two lists, "
                   "least-squares linear regression, and the z-score (standard score) of a value",
    "example_queries": [
        "What is the correlation between [1,2,3] and [2,4,6]?",
        "What is the linear regression of [1,2,3] and [2,4,6]?",
        "What is the z-score of 85 with mean 75 and standard deviation 5?",
    ],
    "keywords": ["correlation", "linear regression", "z-score", "standard score", "least squares"],
    "model_free": True,
}


def precondition(query):
    return bind_statistics(query)[0]


def run(query):
    return dispatch(bind_statistics(query), parse_message="could not parse the statistics query")
