"""CLI for the reasoning compiler.

  capture "<query>"        run the teacher (thinking on), save a reasoning sample
  derive <sample-id>       scaffold an orchestrator module from a sample (then edit it)
  run "<query>"            student loop: recognize → run orchestrator, else teacher fallback
  eval <orchestrator-id>   score an orchestrator's fidelity vs the teacher
  list                     list stored orchestrators
"""

from __future__ import annotations

import argparse
import json
import sys

from . import capture as capmod
from . import derive as derivemod
from . import registry, router
from .teachers import get_teacher


def cmd_capture(args):
    s = capmod.capture(args.query, teacher_backend=args.teacher)
    print(f"captured sample '{s['id']}' (teacher={s['teacher']}) -> {s['_path']}")
    print(f"\nreasoning ({len(s['reasoning'])} chars) + answer:\n{s['answer']}\n")
    print(f"next: python -m reasoning.cli derive {s['id']}")


def cmd_derive(args):
    s = capmod.load_sample(args.sample_id)
    if not s:
        sys.exit(f"no sample '{args.sample_id}' (run capture first)")
    res = derivemod.scaffold(
        s, mode=args.mode, pattern=args.pattern, query_type=args.query_type, force=args.force
    )
    print(f"scaffolded ({res.mode} mode) -> {res.path}")
    if res.mode == "steps":
        gap = res.total - res.matched
        print(f"matched {res.matched}/{res.total} atomic steps to skills "
              f"({gap} model step(s) remain)")
        if gap:
            print("→ each remaining model step is a candidate for a new skill in reasoning/skills/")
    print(f"edit META/run(), then: python -m reasoning.cli eval {s['id']}")


def cmd_run(args):
    from .compiled import model_calls, needs_input
    matcher = {
        "classify": router.ClassifyMatcher,
        "embedding": router.EmbeddingMatcher,
    }.get(args.matcher, router.LexicalMatcher)()
    query = args.query
    m = router.recognize(query, matcher=matcher)
    if not m:
        # No direct route. Try the coded fallbacks cheapest-first before the LLM teacher:
        #   (a) 0-call compound split on conjunctions
        #   (b) ≤2-call single-question reformulation (rewrite → coded answer → verify)
        #   (c) ≤1-call model-assisted compound split
        # Each yields a code-computed answer or None; the teacher is the last resort.
        from . import engine
        from .decompose import decompose_code_first, decompose_with_model
        from .reformulate import reformulate_with_model
        composed = decompose_code_first(query)
        if composed is not None:
            print("[no single match — split into coded sub-answers]\n")
        else:
            composed = reformulate_with_model(query, shot=engine.one_shot)
            if composed is not None:
                print("[no direct match — reformulated into a coded query]\n")
            else:
                composed = decompose_with_model(query, shot=engine.one_shot)
                if composed is not None:
                    print("[no single match — decomposed into coded sub-answers]\n")
        if composed is not None:
            print(composed.answer)
            print(f"\n[coded fallback used {model_calls(composed)} model call(s); "
                  "the answer itself was computed by code]")
            return
        print(f"[no orchestrator matched — falling back to TEACHER ({args.teacher})]\n")
        print(get_teacher(args.teacher).trace(query).answer)
        print('\nTo compile this query type: capture "<query>" → derive → eval.')
        return
    print(f"[recognized: {m.orchestrator.query_type} via {m.via} "
          f"score={m.score:.2f}] running orchestrator — thinking OFF\n")
    r = m.orchestrator.run(query)
    # interactive clarification: the type is recognized but a field is missing — ask, then
    # re-run by APPENDING the answer in a form the bind already parses (still 0 model calls).
    for _ in range(6):
        missing = needs_input(r)
        if not missing:
            break
        print("I need a bit more information:")
        try:
            for _field, (question, template) in missing.items():
                val = input(f"  {question} ").strip()
                if not val:
                    print("\n(cancelled — missing information)")
                    return
                query = f"{query} {template.format(val)}"
        except EOFError:
            print("\n(no input available to clarify)")
            return
        m = router.recognize(query, matcher=matcher) or m
        r = m.orchestrator.run(query)
        print()
    print(r.answer)
    n = model_calls(r)
    if n == 0:
        print("\n[answer used 0 model calls — sidecar not contacted]")
    elif n is None:
        print("\n[pattern orchestrator — model-call count not tracked]")
    else:
        print(f"\n[answer used {n} model call(s)]")
    if args.trace:
        print("\n--- trace ---")
        print(json.dumps(r.trace, indent=2, default=str)[:3000])


def cmd_eval(args):
    o = registry.get(args.orchestrator_id)
    if not o:
        sys.exit(f"no orchestrator '{args.orchestrator_id}' (see: list)")
    from .eval import fidelity

    q = args.query or (o.example_queries[0] if o.example_queries else None)
    if not q:
        sys.exit("no query given and the orchestrator has no example_queries")
    res = fidelity(q, o, teacher_backend=args.teacher)
    nm = "✓" if res["numeric_match"] else ("n/a" if res["orchestrator"] and not res["score"] else "✗")
    print(f"query: {q}\nfidelity: {res['score']}/100 at {res['model_calls']} model call(s) "
          f"(teacher={args.teacher}, numeric_match={nm})\n")
    print(f"--- teacher ({args.teacher}) ---\n{res['teacher']}\n")
    print(f"--- orchestrator (thinking off) ---\n{res['orchestrator']}")


def cmd_eval_all(args):
    from .eval import batch_fidelity

    results = batch_fidelity(teacher_backend=args.teacher)
    agree = sum(r["agrees"] for r in results)
    free = sum(r["model_calls"] == 0 for r in results)
    print(f"Fidelity vs {args.teacher} teacher — {len(results)} coded orchestrators\n")
    print(f"{'orchestrator':24} {'mc':>3} {'ok':>3}  {'coded answer':16} teacher")
    print("-" * 92)
    for r in sorted(results, key=lambda r: r["agrees"]):  # disagreements first
        print(f"{r['id']:24} {r['model_calls']:>3} {'✓' if r['agrees'] else '✗':>3}  "
              f"{r['orchestrator'][:16]:16} {r['teacher']}")
    print("-" * 92)
    print(f"agreement: {agree}/{len(results)}   |   model-free: {free}/{len(results)} "
          f"(every coded answer used 0 model calls)")


def cmd_bench(args):
    from . import bench

    # Reachable-ceiling diagnostic — intrinsic to the data (gold step counts), no routing,
    # so it runs over the FULL slice cheaply. The cumulative curve is the max accuracy a
    # perfect general N-step arithmetic solver could ever reach.
    hist, htot = bench.depth_profile(slice=args.slice)
    print(f"GSM8K [{args.slice}] — arithmetic-depth ceiling (full slice, {htot} questions)\n")
    cum = 0
    cum_label = {"single": "≤1 step", "double": "≤2 steps",
                 "multi(3-4)": "≤4 steps", "deep(5+)": "any depth"}
    for d in bench.DEPTH_ORDER:
        n = hist.get(d, 0)
        if d == "none":
            if n:
                print(f"  {d:12} {n:>5}  ({n / htot:.1%})  — no calc markers (pure-NL / table)")
            continue
        cum += n
        print(f"  {d:12} {n:>5}  ({n / htot:5.1%})   cumulative ceiling {cum / htot:5.1%}"
              f"  ({cum_label[d]})")
    print()

    rep = bench.run(slice=args.slice, limit=args.limit)
    scope = f"{rep.total}" + (f" (--limit)" if args.limit else "")
    print(f"--- code-only engine on {scope} routed questions ---")
    print(f"solved @0call:  {rep.solved_0call}/{rep.total}  ({rep.accuracy:.1%})   ← headline")
    print(f"coverage:       {rep.coverage:.1%} routed to a coded domain\n")
    if rep.no_route_depth:
        print("no_route misses by arithmetic depth (is even the shallow tail unreached?):")
        for d in bench.DEPTH_ORDER:
            n = rep.no_route_depth.get(d, 0)
            if n:
                print(f"  {d:12} {n:>5}")
        print()
    print("miss buckets (a capability work-list):")
    for bucket in ("correct", "wrong_answer", "needs_llm", "no_route"):
        n = rep.buckets.get(bucket, 0)
        if n:
            print(f"  {bucket:14} {n:>5}  ({n / rep.total:.1%})")
    if rep.by_domain:
        print("\ndomains claiming questions (correct / routed):")
        for dom, routed in rep.by_domain.most_common(12):
            print(f"  {dom:22} {rep.domain_correct.get(dom, 0):>4} / {routed:<4}")
    if rep.misses:  # DEV only — TEST keeps none (aggregate-only firewall)
        show = min(args.show_misses, len(rep.misses))
        print(f"\nsample DEV misses (first {show}, for gap-mining):")
        for q, gold, o in rep.misses[:show]:
            tag = "no_route" if not o.matched else (o.domain or "?")
            print(f"  [{tag}] gold={gold}  {q[:90].replace(chr(10), ' ')}")


def cmd_list(args):
    orchs = registry.load_orchestrators()
    if not orchs:
        print("(no orchestrators yet — capture/derive to add one)")
    for o in orchs:
        print(f"{o.id:24} type={o.query_type:22} examples={len(o.example_queries)} kw={o.keywords}")


def cmd_data(args):
    """Manage datasets in $DIFG_DATA_DIR (list / import / describe / info / query / generate /
    convert / gen / domains)."""
    from . import datasets as ds
    a = args.args
    if args.action == "list":
        print("\n".join(ds.list_datasets()) or "(no datasets — import one first)")
    elif args.action == "import":
        print(ds.import_file(a[0], name=args.name))
    elif args.action == "describe":
        print(ds.describe(a[0]))
    elif args.action == "query":
        print(ds.query(a[0], save_as=args.save_as))
    elif args.action == "generate":
        print(ds.generate(int(a[0]), a[1], select=args.select))
    elif args.action == "convert":
        print(ds.convert(a[0], a[1], a[2]))
    elif args.action == "domains":
        from . import datagen
        for dom, gens in datagen.list_domains().items():
            print(f"\n{dom}:")
            for n, doc in gens:
                print(f"  {n:22} {doc}")
        print('\nGenerate with:  data gen <generator> <name> [key=value …]'
              '\n            or:  data gen "generate first 1000 prime numbers"')
    elif args.action == "gen":
        from . import datagen
        if not a:  # list available domain generators
            print("\n".join(f"{k:22} {d}" for k, (_, d) in sorted(datagen.GENERATORS.items())))
            return
        if a[0] in datagen.GENERATORS and len(a) >= 2:           # explicit: <generator> <name> [k=v]
            params = dict(kv.split("=", 1) for kv in a[2:] if "=" in kv)
            print(datagen.generate(a[0], a[1], **params))
            return
        from .bind import bind_dataset                            # natural language: "generate …"
        text = " ".join(a)
        if "generate" not in text.lower() and "synthesize" not in text.lower():
            text = "generate " + text
        ok, d, _, _ = bind_dataset(text)
        if ok and d.get("op") == "generate":
            print(datagen.generate(d["gen"], d["save_as"], **d["params"]))
        else:
            print(f"could not parse a generator from: {' '.join(a)}\n"
                  "try `data domains` to see what's available")
    elif args.action == "info":
        print(ds.describe(a[0]))
        nums = [c for c, t in ds.schema(a[0])
                if any(k in t.upper() for k in ("INT", "DOUBLE", "DEC", "FLOAT", "REAL"))]
        print("\nHow to query it:")
        print(f'  rows:   reasoning.cli data query "SELECT * FROM {a[0]} LIMIT 5"')
        print(f'  count:  reasoning.cli data query "SELECT count(*) FROM {a[0]}"')
        if nums:
            print(f'  agg:    reasoning.cli data query "SELECT avg({nums[0]}) FROM {a[0]}"')
            print(f'  NL:     reasoning.cli run "what is the average of the {nums[0]} '
                  f'column in {a[0]}?"')


def cmd_reference(args):
    """Manage bundled reference data: list tables, import a CSV, review the model-retrieved log,
    or promote a retrieved value into the curated data."""
    from . import reference
    a = args.args
    if args.action == "list":
        print("\n".join(reference.tables()))
    elif args.action == "import":
        print(reference.import_csv(a[0], a[1]))
    elif args.action == "review":
        rows = reference.list_retrieved()
        if not rows:
            print("(retrieved log is empty)")
        for r in rows:
            yr = f" {r['year']}" if r["year"] is not None else ""
            print(f"{r['op']:12} {r['place']}{yr:6} = {r['value']:14} [{r['source']} {r['retrieved_at']}]")
    elif args.action == "promote":
        year = int(a[2]) if len(a) > 2 else None
        print(reference.promote(a[0], a[1], year))


def _print_verify_rounds(res):
    """Render the per-round trace of the author→test→repair loop + the final verdict."""
    for rd in res.get("rounds", []):
        phase = rd.get("phase", "examples")
        tag = "draft" if rd["round"] == 0 else f"repair {rd['round']}"
        if rd.get("skipped"):
            print(f"  [{phase}/{tag}] skipped (not a git repo / no orchestrator)")
            continue
        fails = rd.get("fails") or []
        print(f"  [{phase}/{tag}] " + (f"✓ {phase} pass" if not fails else f"{len(fails)} failing:"))
        for f in fails:
            print(f"      - {f}")
    rtag = {True: "routing✓", False: "routing✗", None: "routing skipped"}[res.get("verified_routing")]
    print(f"  → {'VERIFIED' if res.get('verified') else 'NOT verified'} ({rtag})\n")


def _print_cost(res, model):
    u, cost = res.get("usage"), res.get("cost_usd")
    if not (u or cost is not None):
        return
    # input_tokens is only the *uncached* remainder; the bulk of the prompt is billed as cache, so
    # sum all three for a true prompt-size figure (usage is absent on the extend path → cost only).
    prompt_toks = sum(u.get(k, 0) for k in ("input_tokens", "cache_creation_input_tokens",
                                            "cache_read_input_tokens")) if u else 0
    toks = f" — {prompt_toks} in / {u.get('output_tokens', 0)} out tokens" if u else ""
    print(f"[{model}: {'$%.4f' % cost if cost is not None else 'cost n/a'}{toks}]\n")


def cmd_author_domain(args):
    """Call Claude (high-reasoning) to implement a new domain per the playbook, pre-loaded with
    context. Writes the new module files (never clobbers); wiring + CASES are printed to apply."""
    from . import authoring
    description = " ".join(args.args)
    if not description:
        sys.exit('describe the domain, e.g. author-domain "Thermodynamics: entropy, enthalpy, …"')
    mode = "author+verify loop" if args.verify else "author"
    print(f"[{mode}: '{description[:55]}…' via {args.model} — this calls the model and may take a minute]\n")
    if args.verify:
        res = authoring.author_domain_verified(description, model=args.model, effort=args.effort,
                                               max_rounds=args.max_rounds)
        _print_verify_rounds(res)
    else:
        res = authoring.author_domain(description, model=args.model, effort=args.effort)
    _print_cost(res, args.model)
    files = res["files"]
    if not files:                                  # hard failure — nothing to write
        sys.exit(res["error"] or "no files were produced")
    if res.get("error"):                           # near-miss (e.g. --verify didn't fully pass)
        print(f"⚠ {res['error']} — writing the best-effort draft anyway; fix the unresolved "
              f"failures shown above by hand before committing.\n")
    written, skipped = authoring.write_files(files, dry_run=args.dry_run)
    verb = "would write" if args.dry_run else "wrote"
    print(f"{verb}: {', '.join(written) or '(none)'}")
    if skipped:
        print(f"skipped (already exist — review the generated version manually): {', '.join(skipped)}")
    for meta in ("__wiring__", "__notes__"):
        if meta in files:
            print(f"\n--- {meta.strip('_')} (apply manually) ---\n{files[meta].strip()}")
    print("\nNext: review the files, apply the wiring + CASES, then run "
          "`python -m pytest reasoning/tests reasoning/skills/tests -q` and "
          "`python -m reasoning.cli eval-all`. The model wrote this — verify before committing.")


def cmd_extend_domain(args):
    """Add new skill(s) to an EXISTING domain via Claude: re-emit the domain's three files with the
    new ops folded in, verify (examples + routing — also guards the existing ops), and overwrite."""
    from . import authoring
    description = " ".join(args.args)
    if not description:
        sys.exit('describe the new ops, e.g. extend-domain fluid_mechanics "buoyant force = rho*g*V"')
    verb_mode = "extend" if args.no_verify else "extend+verify loop"
    print(f"[{verb_mode}: {args.domain} += '{description[:45]}…' via {args.model} — calls the model]\n")
    res = authoring.extend_domain(args.domain, description, model=args.model, effort=args.effort,
                                  max_rounds=args.max_rounds, verify=not args.no_verify)
    if res.get("rounds"):
        _print_verify_rounds(res)
    _print_cost(res, args.model)
    files, paths = res["files"], res.get("domain_paths", [])
    if not files or not paths or any(p not in files for p in paths):
        sys.exit(res["error"] or "extend produced no usable files")
    if res.get("error"):
        print(f"⚠ {res['error']} — writing the best-effort update anyway; review `git diff` and fix "
              f"by hand before committing.\n")
    written, _ = authoring.write_files({p: files[p] for p in paths},   # overwrite ONLY the domain's files
                                       dry_run=args.dry_run, overwrite=True)
    print(f"{'would update' if args.dry_run else 'updated'}: {', '.join(written)}")
    if "__wiring__" in files:
        print(f"\n--- new CASES (add to reasoning/tests/test_orchestrators.py) ---\n{files['__wiring__'].strip()}")
    print("\nNext: review `git diff` (the existing ops must be unchanged), add the new CASES, then run "
          "`python -m pytest reasoning/tests reasoning/skills/tests -q`. The model wrote this — verify.")


def cmd_help(args):
    """Introspection: what can I ask? — overview, per-domain examples, or a capability search."""
    from . import introspect
    if args.search:
        hit = introspect.search(args.search)
        print(f"capabilities matching '{args.search}':")
        print("  domains:    " + (", ".join(hit["domains"]) or "(none)"))
        sk = hit["skills"]
        print("  skills:     " + ((", ".join(sk[:20]) + (f" (+{len(sk) - 20} more)" if len(sk) > 20 else ""))
                                   if sk else "(none)"))
        print("  generators: " + (", ".join(hit["generators"]) or "(none)"))
        return
    if args.domain:
        d = introspect.domain(args.domain)
        if not d:
            print(f"no domain '{args.domain}'. Try `reasoning.cli help` or `help --search {args.domain}`.")
            return
        print(f"{d['id']} ({d['type']}) — {d['description']}\nmodel-free: {d['model_free']}\n")
        print("example queries:")
        for e in d["examples"]:
            print(f"  • {e}")
        if d["keywords"]:
            print("\nkeywords: " + ", ".join(d["keywords"]))
        return
    s = introspect.summary()
    print(f"{s['domains']} reasoning domains ({s['model_free_domains']} at 0 model calls), "
          f"{s['skills']} skills, {s['generators']} dataset generators "
          f"({s['generator_domains']} domains).\n")
    for d in introspect.domains():
        print(f"  {d['id']:26} {d['description'][:58]}")
    print("\ndrill in:  reasoning.cli help <domain>   |   search:  reasoning.cli help --search <term>")


def main():
    p = argparse.ArgumentParser(prog="reasoning")
    sub = p.add_subparsers(dest="cmd", required=True)

    teacher_kw = dict(choices=["diffusion", "claude"], default="diffusion",
                      help="teacher backend (default: diffusion self-distillation)")
    c = sub.add_parser("capture"); c.add_argument("query")
    c.add_argument("--teacher", **teacher_kw); c.set_defaults(fn=cmd_capture)
    d = sub.add_parser("derive")
    d.add_argument("sample_id"); d.add_argument("--pattern")
    d.add_argument("--mode", choices=["steps", "pattern"], default=None,
                   help="scaffold style (default: auto — steps when ≥1 line maps to a skill)")
    d.add_argument("--query-type", dest="query_type"); d.add_argument("--force", action="store_true")
    d.set_defaults(fn=cmd_derive)
    r = sub.add_parser("run"); r.add_argument("query")
    r.add_argument("--matcher", choices=["lexical", "classify", "embedding"], default="lexical")
    r.add_argument("--teacher", **teacher_kw)
    r.add_argument("--trace", action="store_true"); r.set_defaults(fn=cmd_run)
    e = sub.add_parser("eval"); e.add_argument("orchestrator_id"); e.add_argument("query", nargs="?")
    e.add_argument("--teacher", **teacher_kw); e.set_defaults(fn=cmd_eval)
    ea = sub.add_parser("eval-all", help="fidelity of every coded orchestrator vs the teacher")
    ea.add_argument("--teacher", choices=["diffusion", "claude"], default="claude",
                    help="teacher backend (default: claude — no sidecar needed)")
    ea.set_defaults(fn=cmd_eval_all)
    bn = sub.add_parser("bench", help="GSM8K-style word-problem accuracy of the code-only engine")
    bn.add_argument("--slice", choices=["dev", "test"], default="dev",
                    help="dev=gsm8k_train (inspectable); test=gsm8k_test (aggregate-only)")
    bn.add_argument("--limit", type=int, default=None, help="cap the number of questions")
    bn.add_argument("--show-misses", dest="show_misses", type=int, default=20,
                    help="how many DEV misses to print for gap-mining")
    bn.set_defaults(fn=cmd_bench)
    sub.add_parser("list").set_defaults(fn=cmd_list)
    da = sub.add_parser("data", help="manage datasets in $DIFG_DATA_DIR")
    da.add_argument("action",
                    choices=["list", "import", "describe", "info", "query", "generate",
                             "convert", "gen", "domains"])
    da.add_argument("args", nargs="*",
                    help="action args (e.g. path/handle/SQL; gen: <generator> <name> key=value … "
                         "or a 'generate …' phrase; info: <handle>)")
    da.add_argument("--name", help="handle name for import")
    da.add_argument("--save-as", dest="save_as", help="save a query result as a new dataset")
    da.add_argument("--select", help="SQL projection for generate (over range(n) t(i))")
    da.set_defaults(fn=cmd_data)
    rf = sub.add_parser("reference", help="manage bundled reference data")
    rf.add_argument("action", choices=["list", "import", "review", "promote"])
    rf.add_argument("args", nargs="*",
                    help="import: <csv> <table>; promote: <op> <place> [year]")
    rf.set_defaults(fn=cmd_reference)
    ad = sub.add_parser("author-domain",
                        help="implement a new domain via Claude, per the playbook")
    ad.add_argument("args", nargs="+", help="the domain description (name + the operations it covers)")
    ad.add_argument("--model", default="haiku",
                    help="Claude model alias/id (default: haiku — cheapest/fastest; "
                         "use --model opus for harder or less-specified domains)")
    ad.add_argument("--effort", choices=["low", "medium", "high", "xhigh", "max"],
                    help="reasoning effort level (default: the CLI default)")
    ad.add_argument("--verify", action="store_true",
                    help="run the author→test→repair loop: check the skill example tables and feed "
                         "failures back to the model until they pass (or --max-rounds is hit)")
    ad.add_argument("--max-rounds", type=int, default=3, help="max repair rounds with --verify (default 3)")
    ad.add_argument("--dry-run", action="store_true", help="print what would be written, don't write")
    ad.set_defaults(fn=cmd_author_domain)
    ed = sub.add_parser("extend-domain", help="add new skill(s) to an existing domain via Claude")
    ed.add_argument("domain", help="the existing domain's module stem (e.g. fluid_mechanics)")
    ed.add_argument("args", nargs="+", help="description of the new operation(s) to add")
    ed.add_argument("--model", default="haiku", help="Claude model alias/id (default: haiku)")
    ed.add_argument("--effort", choices=["low", "medium", "high", "xhigh", "max"],
                    help="reasoning effort level (default: the CLI default)")
    ed.add_argument("--max-rounds", type=int, default=3, help="max repair rounds (default 3)")
    ed.add_argument("--no-verify", action="store_true",
                    help="skip the author→test→repair loop (NOT recommended — it guards that the "
                         "additions don't break the domain's existing ops)")
    ed.add_argument("--dry-run", action="store_true", help="print what would change, don't write")
    ed.set_defaults(fn=cmd_extend_domain)
    hp = sub.add_parser("help", help="introspection: what can I ask? (domains / examples / search)")
    hp.add_argument("domain", nargs="?", help="a domain id to show example queries for")
    hp.add_argument("--search", help="find domains/skills/generators mentioning a term")
    hp.set_defaults(fn=cmd_help)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
