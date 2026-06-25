"""Domain-authoring tool — prompt assembly, file parsing, and write safety. Uses a stub asker;
never calls the real model."""

import importlib
import unittest

from reasoning import authoring as A

duckdb = importlib.util.find_spec("duckdb")


class TestAuthoring(unittest.TestCase):
    def test_build_prompt_embeds_description_and_context(self):
        p = A.build_prompt("Widgets: foo, bar", context="(PLAYBOOK CONTEXT)")
        self.assertIn("Widgets: foo, bar", p)
        self.assertIn("(PLAYBOOK CONTEXT)", p)
        self.assertIn("<<<FILE", p)            # the output-format spec is present
        self.assertIn("num_near", p)           # extraction steering
        self.assertIn("GATE CALIBRATION", p)   # §3f gate-logic steering (allow-list + AND-guard)

    def test_parse_files_roundtrip(self):
        text = ("<<<FILE reasoning/skills/w.py>>>\nX = 1\nY = 2\n<<<END>>>\n"
                "<<<FILE __wiring__>>>\nadd w\n<<<END>>>")
        files = A.parse_files(text)
        self.assertEqual(files["reasoning/skills/w.py"], "X = 1\nY = 2")
        self.assertIn("__wiring__", files)

    def test_author_domain_with_stub(self):
        stub = "<<<FILE reasoning/skills/w.py>>>\nX = 1\n<<<END>>>"
        res = A.author_domain("w", asker=lambda p: stub, context="ctx")
        self.assertIsNone(res["error"])
        self.assertIn("reasoning/skills/w.py", res["files"])
        # usage/cost keys are always present (None on the stub path — no real model call)
        self.assertIn("usage", res)
        self.assertIn("cost_usd", res)
        self.assertIsNone(res["cost_usd"])

    def test_errors_on_empty_or_unparseable(self):
        self.assertIsNotNone(A.author_domain("w", asker=lambda p: None, context="c")["error"])
        self.assertIsNotNone(A.author_domain("w", asker=lambda p: "no blocks", context="c")["error"])

    def test_write_skips_meta_and_refuses_to_clobber(self):
        files = {"reasoning/skills/_authoring_probe_nonexistent.py": "X=1",  # new → would write
                 "__wiring__": "wire",                                       # meta → never a file
                 "reasoning/skills/base.py": "CLOBBER"}                      # exists → skipped
        written, skipped = A.write_files(files, dry_run=True)
        self.assertIn("reasoning/skills/_authoring_probe_nonexistent.py", written)
        self.assertNotIn("__wiring__", written)
        self.assertIn("reasoning/skills/base.py", skipped)

    def test_verified_loop_repairs_a_bad_example(self):
        good = ('<<<FILE reasoning/skills/loopdemo.py>>>\n'
                'from .base import register\n'
                '@register("loopdemo.double", "double x", [((3,), 6)])\n'
                'def double(x):\n    return x * 2\n<<<END>>>')
        bad = good.replace("(3,), 6)", "(3,), 7)")          # wrong expected value
        calls = []

        def stub(p):
            calls.append(p)
            return bad if len(calls) == 1 else good          # draft fails, repair fixes it

        res = A.author_domain_verified("loopdemo", asker=stub, context="ctx", max_rounds=2)
        self.assertTrue(res["verified"], res["rounds"])
        self.assertEqual(len(calls), 2)                      # one draft + one repair
        self.assertTrue(res["rounds"][0]["fails"])           # draft had a failure
        self.assertFalse(res["rounds"][1]["fails"])          # repair passed
        self.assertIn("FAILURES:", calls[1])                 # repair prompt fed the failure back

    def test_verified_loop_gives_up_after_max_rounds(self):
        bad = ('<<<FILE reasoning/skills/loopdemo2.py>>>\n'
               'from .base import register\n'
               '@register("loopdemo2.double", "double x", [((3,), 7)])\n'
               'def double(x):\n    return x * 2\n<<<END>>>')
        res = A.author_domain_verified("d", asker=lambda p: bad, context="c", max_rounds=1)
        self.assertFalse(res["verified"])
        self.assertEqual(len(res["rounds"]), 2)              # draft + 1 repair attempt, both fail

    def test_answer_mismatch_catches_wrong_subskill(self):
        # the acoustics case: query numbers {440, 0.78} match the wave_speed row → must be 343.2
        rows = [(frozenset({440.0, 0.78}), 343.2), (frozenset({343.0, 440.0}), 0.779545)]
        q = "What is the wave speed for a frequency of 440 Hz and wavelength of 0.78 m?"
        self.assertIsNotNone(A._answer_mismatch("1", q, rows))        # wrong sub-skill → flagged
        self.assertIsNone(A._answer_mismatch("343.2", q, rows))       # correct → not flagged
        self.assertIsNone(A._answer_mismatch("343.20001", q, rows))   # within tolerance

    def test_answer_mismatch_skips_when_no_row_matches(self):
        rows = [(frozenset({2.0, 3.0}), 6.0)]
        # query numbers don't exactly match any row → can't verify → no false positive
        self.assertIsNone(A._answer_mismatch("999", "area of radius 5", rows))
        self.assertIsNone(A._answer_mismatch("not numeric", "foo 2 and 3", rows))  # no answer number

    def test_gather_domain_files_locates_existing_domain(self):
        # thermo: skill/bind are thermo.py, but the orchestrator file is thermodynamics.py — found
        # by the "imports bind_thermo" scan, exercising the stem-mismatch fallback.
        files = A.gather_domain_files("thermo")
        self.assertIsNotNone(files)
        self.assertIn("reasoning/skills/thermo.py", files)
        self.assertIn("reasoning/bind/thermo.py", files)
        self.assertTrue(any("orchestrators/thermodynamics.py" in p for p in files))

    def test_gather_domain_files_missing_returns_none(self):
        self.assertIsNone(A.gather_domain_files("no_such_domain_xyz"))

    def test_build_extend_prompt_embeds_files_and_desc(self):
        files = {"reasoning/skills/x.py": "SKILLSRC", "reasoning/bind/x.py": "BINDSRC",
                 "reasoning/store/orchestrators/x.py": "ORCHSRC"}
        p = A.build_extend_prompt("x", "add op foo", files, context="CTX")
        for marker in ("add op foo", "SKILLSRC", "CTX", "PRESERVE every existing op"):
            self.assertIn(marker, p)

    def test_extend_domain_requires_full_reemit(self):
        files0 = A.gather_domain_files("thermo")
        skill_path = next(p for p in files0 if "skills/" in p)
        stub = lambda prompt: f"<<<FILE {skill_path}>>>\nX=1\n<<<END>>>"   # only the skill, no bind/orch
        res = A.extend_domain("thermo", "add op", asker=stub, context="(ctx)", verify=False)
        self.assertIn("missing required files", res["error"])

    def test_extend_domain_stub_no_verify(self):
        files0 = A.gather_domain_files("thermo")
        stub = lambda prompt: "\n".join(f"<<<FILE {p}>>>\n# updated\n<<<END>>>" for p in files0)
        res = A.extend_domain("thermo", "add op", asker=stub, context="(ctx)", verify=False)
        self.assertIsNone(res["error"])
        self.assertEqual(set(res["domain_paths"]), set(files0))

    def test_extend_domain_unknown_domain(self):
        res = A.extend_domain("no_such_domain_xyz", "add op", asker=lambda p: "", context="(ctx)")
        self.assertIn("not found", res["error"])

    def test_write_files_overwrite_mode(self):
        files = {"reasoning/skills/base.py": "X"}                # exists
        self.assertIn("reasoning/skills/base.py",
                      A.write_files(files, dry_run=True, overwrite=True)[0])    # overwrite → written
        self.assertIn("reasoning/skills/base.py",
                      A.write_files(files, dry_run=True)[1])                    # default → skipped

    def test_routing_failures_formatting(self):
        data = {"oid": "widgets",
                "forward": [{"q": "make a widget", "routed": "widgets", "ok": True},
                            {"q": "spin a widget", "routed": "gadgets", "ok": False}],
                "stolen": [{"q": "what is 2+2", "expected": "arithmetic"}]}
        fails = A._routing_failures(data)
        self.assertEqual(len(fails), 2)                       # 1 mis-route + 1 theft (not the ok one)
        self.assertTrue(any("instead of widgets" in f for f in fails))
        self.assertTrue(any("STOLE" in f or "CLAIMS" in f for f in fails))

    def test_verify_routing_skips_without_orchestrator(self):
        # only a skill file (no orchestrator) → routing can't run → None ('skipped'), no git touched
        self.assertIsNone(A._verify_routing({"reasoning/skills/x.py": "X=1"}))

    @unittest.skipUnless(duckdb, "gather_context imports the registry (needs duckdb)")
    def test_gather_context_has_the_essentials(self):
        ctx = A.gather_context()
        for marker in ("THE PLAYBOOK", "guard_residue", "logistics.eoq", "EXISTING query_types"):
            self.assertIn(marker, ctx)


if __name__ == "__main__":
    unittest.main()
