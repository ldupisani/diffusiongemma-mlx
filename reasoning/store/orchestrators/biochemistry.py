"""Biochemistry — GC content, reverse complement, transcription, Henderson–Hasselbalch,
Michaelis–Menten. Coded, zero model calls."""

from reasoning.bind import bind_biochem
from reasoning.compiled import dispatch

META = {
    "id": "biochemistry",
    "query_type": "biochemistry",
    "description": "DNA/RNA operations (GC content, reverse complement, transcription) and "
                   "buffer/enzyme equations (Henderson–Hasselbalch, Michaelis–Menten)",
    "example_queries": [
        "What is the GC content of ATGC?",
        "What is the reverse complement of ATGC?",
        "Transcribe ATGC to mRNA.",
        "What is the buffer pH with pKa 4.76, base 10, and acid 1?",
        "What is the reaction rate with Vmax 100, Km 5, and substrate 15?",
    ],
    "keywords": ["gc content", "reverse complement", "transcribe", "mrna", "henderson",
                 "michaelis", "vmax"],
    "model_free": True,
    "bind_gated": True,
}


def precondition(query):
    return bind_biochem(query)[0]


def run(query):
    return dispatch(bind_biochem(query), parse_message="could not parse the biochemistry inputs")
