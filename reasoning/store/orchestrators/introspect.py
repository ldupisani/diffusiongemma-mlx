"""Self-introspection — answers meta-questions about the system's own capabilities from the
orchestrator/skill/generator registries. Coded, 0 model calls."""
from reasoning import introspect
from reasoning.bind import bind_introspect
from reasoning.compiled import explain
from reasoning.patterns import PatternResult

META = {
    "id": "introspect", "query_type": "introspect",
    "description": "introspection / help: what the system can do — list reasoning domains, what "
                   "you can ask about a topic, and how to query it",
    "example_queries": [
        "What can you do?",
        "What can I ask?",
        "What domains are available?",
        "What kinds of questions can I ask?",
        "What can I ask about geometry?",
        "How do I query physics?",
    ],
    "keywords": ["what can you do", "what can i ask", "capabilities", "what domains",
                 "what topics", "help", "introspect"],
    # bind-gated: a conservative meta-question gate, dispositive across phrasings; a real
    # computational query (which carries digits / a domain match) wins at tier 2 first.
    "model_free": True, "bind_gated": True,
}


def precondition(query):
    return bind_introspect(query)[0]


def _overview():
    s = introspect.summary()
    doms = introspect.domains()
    types = sorted({d["type"] for d in doms})
    lines = [
        f"I answer questions across {s['domains']} reasoning domains "
        f"({s['model_free_domains']} at 0 model calls), backed by {s['skills']} skills and "
        f"{s['generators']} dataset generators in {s['generator_domains']} domains.",
        "",
        "Topics I can reason about:",
        "  " + ", ".join(types),
        "",
        "Ask `What can I ask about <topic>?` to see example queries for one, or use the CLI: "
        "`reasoning.cli help`, `help <domain>`, `help --search <term>`.",
    ]
    return "\n".join(lines)


def _about(topic):
    d = introspect.domain(topic) or introspect.domain(topic.replace(" ", "_"))
    if d:
        out = [f"Domain '{d['id']}' — {d['description']}", "", "Example queries:"]
        out += [f"  • {e}" for e in d["examples"]] or ["  (none listed)"]
        return "\n".join(out)
    hit = introspect.search(topic)
    if not any(hit.values()):
        return (f"I don't have a domain for '{topic}'. Try `What can you do?` for the full list, "
                f"or `reasoning.cli help --search {topic}`.")
    out = [f"Capabilities matching '{topic}':"]
    if hit["domains"]:
        out.append("  domains:    " + ", ".join(hit["domains"]))
    if hit["skills"]:
        sk = hit["skills"]
        out.append(f"  skills:     " + ", ".join(sk[:12]) + (f" (+{len(sk) - 12} more)" if len(sk) > 12 else ""))
    if hit["generators"]:
        out.append("  generators: " + ", ".join(hit["generators"]))
    return "\n".join(out)


def run(query):
    ok, d, _, _ = bind_introspect(query)
    if not ok:
        return explain("could not parse the introspection query")
    ans = _overview() if d["op"] == "overview" else _about(d["topic"])
    return PatternResult(answer=ans,
                         trace=[{"step": f"introspect.{d['op']}", "via": "code"},
                                {"step": "summary", "model_calls": 0}])
