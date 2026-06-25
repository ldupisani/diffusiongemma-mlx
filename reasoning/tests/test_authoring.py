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

    @unittest.skipUnless(duckdb, "gather_context imports the registry (needs duckdb)")
    def test_gather_context_has_the_essentials(self):
        ctx = A.gather_context()
        for marker in ("THE PLAYBOOK", "guard_residue", "logistics.eoq", "EXISTING query_types"):
            self.assertIn(marker, ctx)


if __name__ == "__main__":
    unittest.main()
