"""The MCP service surface — the tool functions answer from code (0 model calls) and the server
registers them. Calls the tool functions directly (they stay plain/callable after registration).
"""

import asyncio
import importlib
import os
import tempfile
import unittest

duckdb = importlib.util.find_spec("duckdb")
mcp = importlib.util.find_spec("mcp")


@unittest.skipUnless(duckdb and mcp, "needs duckdb + mcp")
class TestMCPServer(unittest.TestCase):
    def setUp(self):
        self.data = tempfile.TemporaryDirectory()
        self.ref = tempfile.TemporaryDirectory()
        os.environ["DIFG_DATA_DIR"] = self.data.name
        os.environ["DIFG_REFERENCE_DIR"] = self.ref.name
        from reasoning import mcp_server
        self.M = mcp_server

    def tearDown(self):
        for k in ("DIFG_DATA_DIR", "DIFG_REFERENCE_DIR"):
            os.environ.pop(k, None)
        self.data.cleanup()
        self.ref.cleanup()

    def test_ask_coded_query(self):
        r = self.M.ask("What is the area of a circle with radius 5?")
        self.assertEqual((r["matched"], r["answer"], r["domain"], r["model_calls"]),
                         (True, "78.54", "geometry", 0))

    def test_ask_compound_stays_zero_calls(self):
        r = self.M.ask("What is 2 + 2 and the capital of Japan?")
        self.assertTrue(r["matched"])
        self.assertEqual(r["model_calls"], 0)
        self.assertIn("Tokyo", r["answer"])

    def test_ask_word_problem_no_translation_by_default(self):
        # translator fallback OFF by default → an un-coded word problem yields no answer and is
        # never handled by the translator (no model call)
        r = self.M.ask("Natalia sold 48 clips, then half as many. How many altogether?")
        self.assertIsNone(r["answer"])
        self.assertNotEqual(r.get("domain"), "translate")

    def test_ask_translate_fallback_opt_in(self):
        # with translate=True and a stubbed translator, ask answers via the program path
        from reasoning import translate
        orig = translate.solve
        translate.solve = lambda q, **kw: {"answer": 72.0, "program": "a=48\nanswer=a+a/2",
                                           "model_calls": 1, "error": None}
        try:
            r = self.M.ask("Natalia sold 48 clips, then half as many. How many altogether?",
                           translate=True)
        finally:
            translate.solve = orig
        self.assertEqual((r["matched"], r["answer"], r["domain"], r["model_calls"]),
                         (True, "72", "translate", 1))
        self.assertIn("answer=a+a/2", r["program"])
        self.assertNotIn("low_confidence", r)   # all givens used → no flag

    def test_ask_translate_flags_low_confidence_on_unused_given(self):
        from reasoning import translate
        orig = translate.solve
        translate.solve = lambda q, **kw: {"answer": 49, "program": "answer = 26+15+24-16",
                                           "model_calls": 1, "error": None, "unused_givens": [6.0]}
        try:
            r = self.M.ask("…26…15…24…4…6…", translate=True)
        finally:
            translate.solve = orig
        self.assertTrue(r["low_confidence"])
        self.assertEqual(r["unused_givens"], [6.0])
        self.assertIn("may be wrong", r["warning"])

    def test_ask_unmatched_is_graceful(self):
        r = self.M.ask("Write me a poem about the sea.")
        self.assertFalse(r["matched"])
        self.assertIn("capabilities", r["hint"])

    def test_diagnose_solved_word_problem(self):
        # the model-free reader solves a cross-sentence problem at 0 calls and reports its facts
        r = self.M.diagnose("Tom has 3 apples. Mary has twice as many apples as Tom. "
                            "How many apples do they have together?")
        self.assertEqual((r["solved"], r["answer"], r["model_calls"]), (True, 9, 0))
        self.assertTrue(any("mary" in f for f in r["known"]))
        self.assertNotIn("deficiency", r)

    def test_diagnose_names_missing_world_knowledge(self):
        # a part-count question the corpus recognises → an unsatisfied requirement + a missing-knowledge
        # deficiency phrased as a question (query rather than assume)
        r = self.M.diagnose("What is the total number of legs that Camden's 3 dogs and 2 cats have?")
        self.assertFalse(r["solved"])
        self.assertIsNone(r["answer"])
        self.assertEqual(r["requirements"][0]["shape"], "part-count")
        self.assertFalse(r["requirements"][0]["satisfied"])
        self.assertEqual(r["deficiency"]["kind"], "missing-knowledge")

    def test_diagnose_names_ambiguity(self):
        r = self.M.diagnose("Sabrina gave 10 cookies to her brother. Then Sabrina gave two-thirds of "
                            "her cookies to her sister. If Sabrina had 20 cookies at the start, how "
                            "many cookies are left?")
        self.assertFalse(r["solved"])
        self.assertEqual(r["deficiency"]["kind"], "ambiguous-base")

    def test_ask_attaches_reader_deficiency_when_uncoded(self):
        # an un-coded word problem: ask keeps its precision contract (no reader answer) but surfaces the
        # reader's deficiency so the caller gets a clarifying question, not a bare miss
        r = self.M.ask("What is the total number of legs that Camden's 3 dogs and 2 cats have?")
        self.assertIsNone(r["answer"])
        self.assertEqual(r.get("deficiency", {}).get("kind"), "missing-knowledge")

    def test_ask_coded_query_has_no_deficiency(self):
        # the enrichment must never touch a successfully-coded answer
        r = self.M.ask("What is the area of a circle with radius 5?")
        self.assertEqual(r["answer"], "78.54")
        self.assertNotIn("deficiency", r)

    def test_capabilities_and_describe(self):
        self.assertGreater(self.M.capabilities()["summary"]["domains"], 50)
        self.assertTrue(self.M.describe_domain("geometry")["examples"])
        self.assertIn("error", self.M.describe_domain("no_such_domain"))
        self.assertIn("number_theory", self.M.search_capabilities("prime")["domains"])

    def test_dataset_tools(self):
        self.M.generate_dataset("generate the first 1000 prime numbers")
        self.assertIn("primes", self.M.list_datasets()["datasets"])
        self.assertEqual(self.M.query_dataset("SELECT max(prime) FROM primes"), "7919")
        self.assertEqual([c["name"] for c in self.M.dataset_info("primes")["columns"]],
                         ["index", "prime"])
        self.assertIn("countries", self.M.list_datasets()["reference_tables"])

    def test_tools_registered(self):
        names = sorted(t.name for t in asyncio.run(self.M.mcp.list_tools()))
        for t in ("ask", "capabilities", "describe_domain", "search_capabilities",
                  "list_generators", "generate_dataset", "list_datasets", "dataset_info",
                  "query_dataset"):
            self.assertIn(t, names)

    def test_prompts_registered_and_render(self):
        prompts = {p.name for p in asyncio.run(self.M.mcp.list_prompts())}
        self.assertEqual(prompts, {"ask", "capabilities", "query"})
        r = asyncio.run(self.M.mcp.get_prompt("ask", {"query": "What is 2+2?"}))
        self.assertIn("What is 2+2?", r.messages[0].content.text)
        # the prompt names must not have shadowed the directly-callable tool functions
        self.assertEqual(self.M.ask("What is 2 + 2?")["answer"], "4")


if __name__ == "__main__":
    unittest.main()
