"""Backend layer: selection, per-backend params, capability flags, and tool-call parsers.

These never load a model — they exercise only the pure logic (get_backend constructs an
unloaded backend; gen_params/params_from_request/flags and the tools_util parsers are all
model-free), so the suite stays fast and offline.
"""

import os
import unittest

import backend
import tools_util


class TestBackendSelection(unittest.TestCase):
    def test_auto_detect_from_model_id(self):
        cases = [
            ("mlx-community/Qwen2.5-3B-Instruct-4bit", "autoregressive"),
            ("mlx-community/Llama-3.2-3B-Instruct-4bit", "autoregressive"),
            ("mlx-community/diffusiongemma-26B-A4B-it-4bit", "diffusion"),
            ("some/diffusion-gemma-variant", "diffusion"),
        ]
        for model_id, expected in cases:
            with self.subTest(model_id=model_id):
                self.assertEqual(backend.get_backend(model_id).name, expected)

    def test_explicit_name_overrides_autodetect(self):
        # A diffusion model id forced to the AR backend, and vice versa.
        self.assertEqual(
            backend.get_backend("mlx-community/diffusiongemma-26B-A4B-it-4bit",
                                name="autoregressive").name,
            "autoregressive",
        )
        self.assertEqual(
            backend.get_backend("mlx-community/Qwen2.5-3B-Instruct-4bit",
                                name="diffusion").name,
            "diffusion",
        )

    def test_env_var_overrides_autodetect(self):
        prev = os.environ.get("DIFG_BACKEND")
        os.environ["DIFG_BACKEND"] = "diffusion"
        try:
            # Qwen id auto-detects autoregressive, but the env var wins.
            self.assertEqual(
                backend.get_backend("mlx-community/Qwen2.5-3B-Instruct-4bit").name,
                "diffusion",
            )
        finally:
            if prev is None:
                del os.environ["DIFG_BACKEND"]
            else:
                os.environ["DIFG_BACKEND"] = prev

    def test_explicit_name_beats_env_var(self):
        prev = os.environ.get("DIFG_BACKEND")
        os.environ["DIFG_BACKEND"] = "diffusion"
        try:
            self.assertEqual(
                backend.get_backend("any/model", name="autoregressive").name,
                "autoregressive",
            )
        finally:
            if prev is None:
                del os.environ["DIFG_BACKEND"]
            else:
                os.environ["DIFG_BACKEND"] = prev

    def test_unknown_backend_name_raises(self):
        with self.assertRaises(ValueError):
            backend.get_backend("any/model", name="bogus")

    def test_aliases_resolve_to_autoregressive(self):
        for alias in ("ar", "mlx-lm", "mlx_lm", "autoregressive"):
            with self.subTest(alias=alias):
                self.assertEqual(backend.get_backend("any/model", name=alias).name,
                                 "autoregressive")


class TestCapabilityFlags(unittest.TestCase):
    def test_autoregressive_flags(self):
        b = backend.get_backend("mlx-community/Qwen2.5-3B-Instruct-4bit")
        self.assertFalse(b.supports_thinking)
        self.assertEqual(b.tool_call_format, "json")

    def test_diffusion_flags(self):
        b = backend.get_backend("mlx-community/diffusiongemma-26B-A4B-it-4bit")
        self.assertTrue(b.supports_thinking)
        self.assertEqual(b.tool_call_format, "gemma-dsl")


class TestGenParams(unittest.TestCase):
    def setUp(self):
        self.ar = backend.get_backend("mlx-community/Qwen2.5-3B-Instruct-4bit")
        self.df = backend.get_backend("mlx-community/diffusiongemma-26B-A4B-it-4bit")

    def test_autoregressive_gen_params_is_temperature_only(self):
        for quality, temp in [("fast", 0.0), ("balanced", 0.4), ("quality", 0.7)]:
            with self.subTest(quality=quality):
                self.assertEqual(self.ar.gen_params(quality, 256), {"temperature": temp})

    def test_diffusion_gen_params_has_canvas_knobs(self):
        p = self.df.gen_params("balanced", 256)
        self.assertEqual(p["max_denoising_steps"], 32)
        self.assertEqual(p["diffusion_sampler"], "entropy-bound")
        self.assertEqual(p["block_length"], 256)
        self.assertIn("temperature", p)

    def test_diffusion_block_length_floors_at_256(self):
        self.assertEqual(self.df.gen_params("fast", 64)["block_length"], 256)
        self.assertEqual(self.df.gen_params("fast", 512)["block_length"], 512)

    def test_params_from_request_autoregressive_pulls_only_temperature(self):
        req = {"temperature": 0.9, "max_denoising_steps": 20, "block_length": 300}
        self.assertEqual(self.ar.params_from_request(req), {"temperature": 0.9})

    def test_params_from_request_autoregressive_defaults_temperature(self):
        self.assertEqual(self.ar.params_from_request({}), {"temperature": 0.4})

    def test_params_from_request_diffusion_honors_client_values(self):
        req = {"temperature": 0.9, "max_denoising_steps": 20,
               "block_length": 300, "diffusion_sampler": "confidence-threshold"}
        self.assertEqual(self.df.params_from_request(req), {
            "temperature": 0.9, "max_denoising_steps": 20,
            "block_length": 300, "diffusion_sampler": "confidence-threshold",
        })

    def test_params_from_request_diffusion_defaults(self):
        p = self.df.params_from_request({})
        self.assertEqual(p, {"temperature": 0.4, "max_denoising_steps": 32,
                             "block_length": 256, "diffusion_sampler": "entropy-bound"})


class TestContextLimit(unittest.TestCase):
    def test_flat_dict(self):
        self.assertEqual(backend.context_limit({"max_position_embeddings": 4096}), 4096)

    def test_attr_object(self):
        cfg = type("Cfg", (), {"max_position_embeddings": 8192})()
        self.assertEqual(backend.context_limit(cfg), 8192)

    def test_nested_text_config_wins(self):
        cfg = type("Cfg", (), {"text_config": {"max_position_embeddings": 1234}})()
        self.assertEqual(backend.context_limit(cfg), 1234)

    def test_missing_returns_none(self):
        self.assertIsNone(backend.context_limit({}))


class TestParseToolCallsJson(unittest.TestCase):
    """Qwen/Hermes JSON tool-call format: <tool_call>{...}</tool_call>."""

    def test_single_call(self):
        text = 'sure <tool_call>{"name":"add","arguments":{"a":1,"b":2}}</tool_call> done'
        self.assertEqual(tools_util.parse_tool_calls_json(text), [("add", {"a": 1, "b": 2})])

    def test_parallel_calls(self):
        text = ('<tool_call>{"name":"a","arguments":{"x":1}}</tool_call>'
                '<tool_call>{"name":"b","arguments":{"y":2}}</tool_call>')
        self.assertEqual(tools_util.parse_tool_calls_json(text),
                         [("a", {"x": 1}), ("b", {"y": 2})])

    def test_missing_terminator(self):
        # No closing </tool_call>: fall back to the balanced JSON object after the opener.
        text = '<tool_call>{"name":"f","arguments":{"q":"hi"}}'
        self.assertEqual(tools_util.parse_tool_calls_json(text), [("f", {"q": "hi"})])

    def test_arguments_as_json_string(self):
        text = '<tool_call>{"name":"g","arguments":"{\\"z\\":3}"}</tool_call>'
        self.assertEqual(tools_util.parse_tool_calls_json(text), [("g", {"z": 3})])

    def test_doubled_braces_qwen_template_quirk(self):
        # Qwen2.5's bundled chat template shows the example as {{...}}; the model copies it.
        text = '<tool_call>{{"name": "add", "arguments": {"a": 21, "b": 21}}}}\n<tool_call>'
        self.assertEqual(tools_util.parse_tool_calls_json(text), [("add", {"a": 21, "b": 21})])

    def test_no_calls(self):
        self.assertEqual(tools_util.parse_tool_calls_json("just a plain answer"), [])

    def test_missing_name_key_skipped(self):
        text = '<tool_call>{"arguments":{"a":1}}</tool_call>'
        self.assertEqual(tools_util.parse_tool_calls_json(text), [])

    def test_missing_arguments_defaults_empty(self):
        text = '<tool_call>{"name":"ping"}</tool_call>'
        self.assertEqual(tools_util.parse_tool_calls_json(text), [("ping", {})])


class TestParseToolCallsGemma(unittest.TestCase):
    """Gemma DSL: <|tool_call>call:NAME{...}<tool_call|>."""

    def test_single_call(self):
        self.assertEqual(tools_util.parse_tool_calls('<|tool_call>call:add{a:1,b:2}<tool_call|>'),
                         [("add", {"a": 1, "b": 2})])

    def test_quoted_string_arg(self):
        text = '<|tool_call>call:greet{name:<|"|>Ada<|"|>}<tool_call|>'
        self.assertEqual(tools_util.parse_tool_calls(text), [("greet", {"name": "Ada"})])

    def test_parallel_calls_without_terminator(self):
        text = '<|tool_call>call:a{x:1}<|tool_call>call:b{y:2}'
        self.assertEqual(tools_util.parse_tool_calls(text), [("a", {"x": 1}), ("b", {"y": 2})])

    def test_no_calls(self):
        self.assertEqual(tools_util.parse_tool_calls("plain text"), [])


class TestStripThoughtChannel(unittest.TestCase):
    def test_strips_thought_block(self):
        self.assertEqual(
            tools_util.strip_thought_channel("<|channel>thought reasoning here<channel|>  answer"),
            "answer",
        )

    def test_no_thought_is_unchanged(self):
        self.assertEqual(tools_util.strip_thought_channel("  just an answer  "), "just an answer")


if __name__ == "__main__":
    unittest.main()
