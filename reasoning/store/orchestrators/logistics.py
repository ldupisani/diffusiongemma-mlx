"""Logistics / operations research — EOQ, reorder point, safety stock, inventory turnover,
days of supply, Little's Law, trucks needed, fuel cost. Coded, 0 calls."""
from reasoning.bind import bind_logistics
from reasoning.compiled import dispatch

META = {
    "id": "logistics", "query_type": "logistics",
    "description": "logistics & operations research: Economic Order Quantity, reorder point, "
                   "safety stock, inventory turnover, days of supply, Little's Law, the number "
                   "of trucks/containers needed, and trip fuel cost",
    "example_queries": [
        "What is the EOQ for an annual demand of 1000, order cost of 10, and holding cost of 2?",
        "What is the reorder point for a demand of 50 per day and a lead time of 5 days?",
        "What is the safety stock for z 1.65, standard deviation 20, and lead time 4?",
        "What is the inventory turnover for COGS of 500000 and average inventory of 100000?",
        "How many trucks are needed to ship 1000 units with a capacity of 150 each?",
        "What is the fuel cost for a 600 km trip at 12 km/l and $1.50 per liter?",
        "What is the inventory using Little's Law for a throughput of 10 and a flow time of 5?",
        "What are the days of supply for an inventory of 500 and daily demand of 50?",
    ],
    "keywords": ["eoq", "reorder point", "safety stock", "inventory turnover", "logistics",
                 "trucks needed", "fuel cost", "little's law", "days of supply"],
    "model_free": True,
}


def precondition(query):
    return bind_logistics(query)[0]


def run(query):
    return dispatch(bind_logistics(query), parse_message="could not parse the logistics query")
