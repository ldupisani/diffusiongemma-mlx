"""Machine learning theory — losses, activations, classifier metrics, impurity, gradient step.
Coded, 0 calls."""
from reasoning.bind import bind_ml
from reasoning.compiled import dispatch

META = {
    "id": "ml_theory", "query_type": "ml_theory",
    "description": "machine-learning quantities: sigmoid/ReLU/softmax, MSE and cross-entropy "
                   "losses, accuracy/precision/recall/F1, Shannon entropy, Gini impurity, "
                   "and a gradient-descent step",
    "example_queries": [
        "What is the sigmoid of 2?",
        "What is the softmax of [1, 2, 3]?",
        "What is the F1 score with tp=8, fp=2, fn=1?",
        "What is the Shannon entropy of [0.5, 0.5]?",
    ],
    "keywords": ["sigmoid", "softmax", "cross-entropy", "f1 score", "precision", "gini"],
    "model_free": True,
}


def precondition(query):
    return bind_ml(query)[0]


def run(query):
    return dispatch(bind_ml(query), parse_message="could not parse the machine-learning query")
