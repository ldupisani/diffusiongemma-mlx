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

    def test_ask_unmatched_is_graceful(self):
        r = self.M.ask("Write me a poem about the sea.")
        self.assertFalse(r["matched"])
        self.assertIn("capabilities", r["hint"])

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
