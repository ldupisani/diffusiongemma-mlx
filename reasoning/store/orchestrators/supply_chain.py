"""Supply chain optimization — total annual inventory cost, Economic Production Quantity,
newsvendor critical ratio, capacity utilization, takt time. Complements (not duplicates)
logistics. Coded, 0 model calls."""
from reasoning.bind import bind_supply_chain
from reasoning.bind._common import guard_residue
from reasoning.compiled import dispatch

META = {
    "id": "supply_chain", "query_type": "supply_chain",
    "description": "supply chain optimization: total annual inventory cost (ordering + holding), "
                   "Economic Production Quantity for constrained production, newsvendor critical "
                   "ratio for optimal stocking, capacity utilization as a percentage, and takt time "
                   "for demand-driven production pacing",
    "example_queries": [
        "What is the total annual inventory cost for demand 1000, order quantity 100, order cost 10, holding cost 2?",
        "What is the EPQ for demand 5000, order cost 50, holding cost 5, production rate 1000, demand rate 200?",
        "What is the newsvendor critical ratio for shortage cost 20 and excess cost 10?",
        "What is the capacity utilization if demand is 800 and capacity is 1000?",
        "What is the takt time for 480 minutes available and 60 customer units demanded?",
    ],
    "keywords": ["supply chain", "total annual cost", "EPQ", "economic production quantity",
                 "newsvendor", "critical ratio", "capacity utilization", "takt time",
                 "shortage cost", "excess cost", "underage", "overage"],
    "model_free": True,
}


def precondition(query):
    return bind_supply_chain(query)[0]


def run(query):
    bound = guard_residue(query, bind_supply_chain(query))
    return dispatch(bound, parse_message="could not parse the supply chain query")