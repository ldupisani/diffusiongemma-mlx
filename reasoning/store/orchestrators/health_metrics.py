"""Health / body metrics — BMI, heart rate, pace, BMR, ideal weight. Coded, zero model."""

from reasoning.bind import bind_health, clarify_health
from reasoning.compiled import clarify, dispatch

META = {
    "id": "health_metrics",
    "query_type": "health_metrics",
    "description": "BMI, maximum/target heart rate, running pace, BMR, or ideal body weight",
    "example_queries": [
        "What is the BMI of a person who weighs 70 kg and is 1.75 meters tall?",
        "What is the maximum heart rate for a 25 year old?",
        "What is the target heart rate at 70 percent intensity for a 40 year old?",
        "What is the pace per mile if you run 5 miles in 40 minutes?",
        "What is the BMR of a 30 year old man weighing 80 kg and 180 cm tall?",
        "What is the ideal body weight for a man who is 180 cm tall?",
    ],
    "keywords": ["bmi", "heart rate", "pace", "bmr", "ideal body weight", "calories"],
    "model_free": True,
}


def precondition(query):
    return bind_health(query)[0]


clarify_probe = clarify_health  # routed only when nothing can ANSWER (a lower router tier)


def run(query):
    ok = bind_health(query)[0]
    if not ok:
        missing = clarify_health(query)
        if missing:
            return clarify(missing)
    return dispatch(bind_health(query), parse_message="could not parse the health inputs")
