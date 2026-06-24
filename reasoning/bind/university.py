"""bind layer — university domains (see reasoning.bind)."""
from __future__ import annotations

import re

from reasoning.bind._common import *  # noqa: F401,F403


# --- set theory: explicit set operations on {…} literals --------------------------------
_BRACE = re.compile(r"\{([^{}]*)\}")


def _parse_set(body: str):
    """A {…} body → frozenset of ints (supports 'a..b' / 'a to b' ranges); None if any
    element isn't an integer (conservative — don't claim a non-numeric set)."""
    body = body.strip()
    if not body:
        return frozenset()
    out = set()
    for tok in body.split(","):
        tok = tok.strip()
        rng = re.fullmatch(r"(-?\d+)\s*(?:\.\.|to|through)\s*(-?\d+)", tok)
        if rng:
            out.update(range(int(rng[1]), int(rng[2]) + 1))
        elif re.fullmatch(r"-?\d+", tok):
            out.add(int(tok))
        else:
            return None
    return frozenset(out)


def bind_set_operations(query: str) -> tuple[bool, dict | None, str, int]:
    q = query.lower()
    sets = [_parse_set(b) for b in _BRACE.findall(query)]
    if not sets or any(s is None for s in sets):
        return False, None, "code", 0

    def ok(skill, kind, args):
        return True, {"skill": skill, "kind": kind, "args": args}, "code", 0

    if "power set" in q:
        return ok("setops.power_set", "set", [sets[0]])
    if "how many subsets" in q or "number of subsets" in q:
        return ok("setops.count_subsets", "int", [sets[0]])
    if len(sets) >= 2:
        a, b = sets[0], sets[1]
        if "cartesian" in q or "cross product" in q or "×" in query:
            return ok("setops.cartesian_product", "set", [a, b])
        if "symmetric difference" in q:
            return ok("setops.symmetric_difference", "set", [a, b])
        if "complement" in q:  # "complement of A in U" → complement(universe=U, a=A)
            return ok("setops.complement", "set", [b, a])
        if "union" in q or "∪" in query:
            return ok("setops.union", "set", [a, b])
        if "intersection" in q or "∩" in query:
            return ok("setops.intersection", "set", [a, b])
        if "difference" in q or "minus" in q or "\\" in query:
            return ok("setops.difference", "set", [a, b])
        if "proper subset" in q:  # before "subset" (substring)
            return ok("setops.is_proper_subset", "bool", [a, b])
        if "superset" in q:
            return ok("setops.is_superset", "bool", [a, b])
        if "subset" in q:
            return ok("setops.is_subset", "bool", [a, b])
        if "disjoint" in q:
            return ok("setops.is_disjoint", "bool", [a, b])
        if "equal" in q or "the same" in q:
            return ok("setops.set_equals", "bool", [a, b])
    if any(p in q for p in ("cardinality", "how many elements", "number of elements",
                            "size of", "cardinal")):
        return ok("setops.cardinality", "int", [sets[0]])
    return False, None, "code", 0


# --- group theory: cyclic groups Z_n, permutations, symmetric group ---------------------
_ZN = re.compile(r"z[\s_/]*?(\d+)|z\s*mod(?:ulo)?\s*(\d+)|integers\s+mod(?:ulo)?\s+(\d+)",
                 re.IGNORECASE)
_PERM_TUPLE = re.compile(r"[\(\[]\s*(\d+(?:\s*[,\s]\s*\d+)+)\s*[\)\]]")
_SYM_GROUP = re.compile(r"\bs[_\s]*(\d+)\b|symmetric group (?:on|of|s_?)?\s*(\d+)", re.IGNORECASE)


def _parse_perm(body: str) -> tuple:
    return tuple(int(t) for t in re.split(r"[,\s]+", body.strip()) if t)


def bind_group_theory(query: str) -> tuple[bool, dict | None, str, int]:
    q = query.lower()
    perms = [_parse_perm(b) for b in _PERM_TUPLE.findall(query)]

    def ok(skill, kind, args):
        return True, {"skill": skill, "kind": kind, "args": args}, "code", 0

    # dihedral group order
    dm = re.search(r"\bd[_\s]*(\d+)\b", query, re.IGNORECASE)
    if "dihedral" in q and dm:
        return ok("group.dihedral_order", "int", [int(dm.group(1))])

    # symmetric group order
    sm = _SYM_GROUP.search(query)
    if sm and ("symmetric group" in q or re.search(r"\bs_?\d+\b", q)) \
            and any(w in q for w in ("order", "size", "how many", "cardinality", "elements")):
        return ok("group.symmetric_group_order", "int", [int(sm.group(1) or sm.group(2))])

    # permutations (require the word "permutation" and a tuple)
    if "permutation" in q and perms:
        if "compose" in q and len(perms) >= 2:
            return ok("group.compose_permutations", "perm", [perms[0], perms[1]])
        if "inverse" in q:
            return ok("group.permutation_inverse", "perm", [perms[0]])
        if "cycle" in q:
            return ok("group.permutation_cycles", "str", [perms[0]])
        if any(w in q for w in ("even", "odd", "parity", "sign")):
            return ok("group.permutation_parity", "str", [perms[0]])
        if "order" in q:
            return ok("group.permutation_order", "int", [perms[0]])

    # cyclic group Z_n
    zn = _ZN.search(query)
    if zn:
        n = int(next(g for g in zn.groups() if g))
        if "subgroup" in q and "generated by" in q:
            m = re.search(r"generated by\s+(\d+)", q)
            if m:
                return ok("group.subgroup_generated", "genset", [int(m.group(1)), n])
        if "generator" in q:
            return ok("group.zn_generators", "genset", [n])
        if "inverse" in q:
            m = re.search(r"inverse of\s+(\d+)", q)
            if m:
                return ok("group.zn_inverse", "int", [int(m.group(1)), n])
        add = re.search(r"(\d+)\s*\+\s*(\d+)", query)
        if add:
            return ok("group.zn_add", "int", [int(add.group(1)), int(add.group(2)), n])
        if "order" in q:
            m = re.search(r"order of (?:the element\s+)?(\d+)", q)
            if m:
                return ok("group.element_order", "int", [int(m.group(1)), n])
    return False, None, "code", 0


# --- topology: finite spaces (X, T) ------------------------------------------------------
def _top_level_braces(s: str) -> list:
    """Each top-level {…} group → (content, is_nested, open_index). Handles nested braces
    (a topology is a set of sets) via balanced matching."""
    out, depth, start, nested = [], 0, None, False
    for i, ch in enumerate(s):
        if ch == "{":
            if depth == 0:
                start, nested = i, False
            else:
                nested = True
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0:
                out.append((s[start + 1:i], nested, start))
    return out


def _parse_topology(content: str):
    """A {∅, {1}, {1,2}} body → frozenset of frozensets (the open sets)."""
    sets = set()
    if "∅" in content or re.search(r"\{\s*\}", content):
        sets.add(frozenset())
    for body in re.findall(r"\{([^{}]*)\}", content):
        s = _parse_set(body)
        if s is None:
            return None
        sets.add(s)
    return frozenset(sets)


def bind_topology(query: str) -> tuple[bool, dict | None, str, int]:
    q = query.lower()
    if "topolog" not in q and "open set" not in q and "closed set" not in q \
            and "hausdorff" not in q and "connected" not in q:
        return False, None, "code", 0
    groups = _top_level_braces(query)
    topo = next(((_parse_topology(c), idx) for c, nested, idx in groups
                 if nested or "∅" in c or re.search(r"\{\s*\}", c)), None)
    if topo is None or topo[0] is None:
        return False, None, "code", 0
    T, _ = topo
    flats = [(_parse_set(c), idx) for c, nested, idx in groups
             if not (nested or "∅" in c or re.search(r"\{\s*\}", c))]
    flats = [(s, idx) for s, idx in flats if s is not None]
    on = q.rfind(" on ")
    X = next((s for s, idx in flats if on != -1 and idx > on), None)
    if X is None:  # no explicit "on {X}" → X is the largest open set (∪T)
        X = frozenset().union(*T) if T else frozenset()
    A = next((s for s, idx in flats if s is not X), None)

    def ok(skill, kind, args):
        return True, {"skill": skill, "kind": kind, "args": args}, "code", 0

    if "is" in q and "a topology" in q:
        return ok("topology.is_topology", "bool", [X, T])
    if "connected" in q:
        return ok("topology.is_connected", "bool", [X, T])
    if "hausdorff" in q or "t2" in q:
        return ok("topology.is_hausdorff", "bool", [X, T])
    if re.search(r"\bt1\b", q):
        return ok("topology.is_t1", "bool", [X, T])
    if re.search(r"\bt0\b", q):
        return ok("topology.is_t0", "bool", [X, T])
    if A is not None:
        if "exterior" in q:
            return ok("topology.exterior", "set", [X, T, A])
        if "dense" in q:
            return ok("topology.is_dense", "bool", [X, T, A])
        if "closure" in q:
            return ok("topology.closure", "set", [X, T, A])
        if "interior" in q:
            return ok("topology.interior", "set", [X, T, A])
        if "boundary" in q:
            return ok("topology.boundary", "set", [X, T, A])
        if "open" in q:
            return ok("topology.is_open", "bool", [T, A])
        if "closed" in q:
            return ok("topology.is_closed", "bool", [X, T, A])
    return False, None, "code", 0


# --- algebra & calculus (SymPy-backed): parse the expression, validate it actually parses --
def _valid_math(expr: str) -> bool:
    if not re.search(r"[a-zA-Z]", expr):  # must contain a variable
        return False
    try:
        from reasoning.skills.algebra import parse
        parse(expr.split("=")[0])  # validate the LHS (solve may carry "= rhs")
        return True
    except Exception:
        return False


_ALG_KW = [("solve", "algebra.solve"), ("factorize", "algebra.factor"),
           ("factorise", "algebra.factor"), ("factor", "algebra.factor"),
           ("expand", "algebra.expand"), ("simplify", "algebra.simplify")]


def bind_algebra(query: str) -> tuple[bool, dict | None, str, int]:
    q = query.lower()

    # ω / "ordinal" means this is ordinal arithmetic, not symbolic algebra (sympy would parse
    # the words as variables) — decline so bind_ordinal claims it.
    if "ω" in query or "omega" in q or "ordinal" in q:
        return False, None, "code", 0

    # solve a SYSTEM of equations (before plain "solve")
    if "system" in q:
        em = re.search(r"system\s*(?:of\s+equations)?\s*:?\s*(.+?)\s*[?.]?$", query, re.IGNORECASE)
        if em:
            clauses = [c.strip() for c in re.split(r"\s+and\s+|;", em.group(1)) if "=" in c]
            if len(clauses) >= 2:
                varset = sorted(set(re.findall(r"\b([a-z])\b", " ".join(clauses))))
                if varset and all(_valid_math(c.split("=")[0]) for c in clauses):
                    return True, {"skill": "algebra.solve_system",
                                  "args": ["; ".join(clauses), ", ".join(varset)]}, "code", 0
        return False, None, "code", 0

    # evaluate EXPR at x = value
    if "evaluate" in q or "value of" in q:
        em = re.search(r"(?:evaluate|value of)\s+(.+?)\s+(?:at|when|for)\s+([a-z])\s*=\s*"
                       r"([^\s?.,]+)", query, re.IGNORECASE)
        if em and _valid_math(em.group(1)):
            return True, {"skill": "algebra.evaluate",
                          "args": [em.group(1).strip(), em.group(2), em.group(3)]}, "code", 0
        return False, None, "code", 0

    # degree of a polynomial
    if "degree" in q:
        em = re.search(r"degree of\s+(?:the\s+)?(?:polynomial\s+)?(.+?)\s*[?.]?$",
                       query, re.IGNORECASE)
        if em and _valid_math(em.group(1)):
            return True, {"skill": "algebra.degree",
                          "args": [em.group(1).strip(), "x"]}, "code", 0
        return False, None, "code", 0

    skill = next((sk for kw, sk in _ALG_KW if kw in q), None)
    if not skill:
        return False, None, "code", 0
    m = re.search(r"(?:solve|factori[sz]e|factor|expand|simplify)\s+(.+)$", query, re.IGNORECASE)
    if not m:
        return False, None, "code", 0
    rest, var = m.group(1).strip(), "x"
    # the solve variable must be a SINGLE letter at the end ("for x"); "for z-1" is not a
    # valid variable — decline rather than silently grabbing "z" and dropping "-1".
    vm = re.search(r"\bfor\s+([a-z])\s*[?.]?\s*$", rest, re.IGNORECASE)
    if vm:
        var, rest = vm.group(1), rest[:vm.start()]
    elif re.search(r"\bfor\b", rest, re.IGNORECASE):
        return False, None, "code", 0
    expr = re.sub(r"^(?:the|expression|equation|polynomial|function|of)\s+", "", rest.strip(),
                  flags=re.IGNORECASE).strip().rstrip("?.").strip()
    if not _valid_math(expr):
        return False, None, "code", 0
    args = [expr, var] if skill == "algebra.solve" else [expr]
    return True, {"skill": skill, "args": args}, "code", 0


def bind_calculus(query: str) -> tuple[bool, dict | None, str, int]:
    q = query.lower()
    var = "x"
    vm = re.search(r"with respect to\s+([a-z])", q) or re.search(r"\bd\s*/\s*d([a-z])\b", q)
    if vm:
        var = vm.group(1)

    def fin(skill, args):
        if not _valid_math(args[0]):
            return False, None, "code", 0
        return True, {"skill": skill, "args": args}, "code", 0

    if "critical point" in q:
        em = re.search(r"critical points? of\s+(.+?)\s*[?.]?$", query, re.IGNORECASE)
        if em:
            return fin("calculus.critical_points", [em.group(1).strip(), var])
        return False, None, "code", 0
    if "taylor" in q or "maclaurin" in q:
        em = re.search(r"(?:taylor|maclaurin) series of\s+(.+?)\s+(?:around|about|at|to|up)",
                       query, re.IGNORECASE)
        order = re.search(r"order\s+(\d+)", q)
        pm = re.search(r"\b(?:around|about|at)\s+([^\s?.,]+)", query, re.IGNORECASE)
        if em and order:
            pt = "0" if not pm or "maclaurin" in q else pm.group(1)
            return fin("calculus.taylor_series",
                       [em.group(1).strip(), var, pt, int(order.group(1))])
        return False, None, "code", 0
    if "limit" in q:
        em = re.search(r"limit of\s+(.+?)\s+as\b", query, re.IGNORECASE)
        am = re.search(r"as\s+([a-z])\s*(?:->|→|approaches|goes to|tends to)\s*([^\s?.,]+)",
                       query, re.IGNORECASE)
        if em and am:
            pt = am.group(2)
            pt = "oo" if ("inf" in pt.lower() or pt in ("∞",)) else pt
            return fin("calculus.limit", [em.group(1).strip(), am.group(1), pt])
        return False, None, "code", 0
    if any(w in q for w in ("integral", "integrate", "antiderivative")):
        em = re.search(r"(?:integral of|integrate|antiderivative of)\s+(.+?)"
                       r"(?:\s+from\b|\s+with respect|\s+d[a-z]\b|\s+for\b|[?.]|$)",
                       query, re.IGNORECASE)
        fm = re.search(r"from\s+(\S+?)\s+to\s+([^\s?.,]+)", query, re.IGNORECASE)
        if em:
            expr = em.group(1).strip()
            if fm:
                return fin("calculus.definite_integral", [expr, var, fm.group(1), fm.group(2)])
            return fin("calculus.integral", [expr, var])
        return False, None, "code", 0
    if "derivative" in q or "differentiate" in q:
        em = re.search(r"(?:derivative of|differentiate)\s+(.+?)"
                       r"(?:\s+with respect|\s+for\b|[?.]|$)", query, re.IGNORECASE)
        if em:
            expr = em.group(1).strip()
            ordinals = {"second": 2, "third": 3, "fourth": 4, "fifth": 5}
            om = re.search(r"\b(\d+)(?:st|nd|rd|th)\s+derivative", q)
            n = (int(om.group(1)) if om
                 else next((v for w, v in ordinals.items() if f"{w} derivative" in q), None))
            if n and n >= 2:
                return fin("calculus.nth_derivative", [expr, var, n])
            return fin("calculus.derivative", [expr, var])
    return False, None, "code", 0
