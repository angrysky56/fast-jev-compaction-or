"""Contract tests runnable without installing Hermes Agent."""

from __future__ import annotations

import copy
import importlib.util
import os
import sys
import types
import unittest
from pathlib import Path


def load_engine():
    agent = types.ModuleType("agent")
    context_engine = types.ModuleType("agent.context_engine")

    class ContextEngine:  # noqa: D101 - test double for Hermes' abstract base
        pass

    context_engine.ContextEngine = ContextEngine
    sys.modules.setdefault("agent", agent)
    sys.modules["agent.context_engine"] = context_engine
    path = Path(__file__).parents[1] / "context_engine" / "fast_jev_compaction" / "__init__.py"
    spec = importlib.util.spec_from_file_location("fast_jev_compaction_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ENGINE_MODULE = load_engine()
Engine = ENGINE_MODULE.FastJevCompactionEngine


def read_pair(call_id: str, name: str = "Read"):
    return [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": call_id, "function": {"name": name, "arguments": '{"path":"a.py"}'}},
            ],
        },
        {"role": "tool", "tool_call_id": call_id, "content": "x" * 800},
    ]


class ContextEngineTests(unittest.TestCase):
    def setUp(self):
        self.previous_key = os.environ.get("OPENROUTER_API_KEY")
        os.environ["OPENROUTER_API_KEY"] = "test-key"

    def tearDown(self):
        if self.previous_key is None:
            os.environ.pop("OPENROUTER_API_KEY", None)
        else:
            os.environ["OPENROUTER_API_KEY"] = self.previous_key

    def test_compress_truncates_a_scored_result_and_keeps_edits(self):
        engine = Engine()
        engine.protect_first_n = 0
        engine.protect_last_n = 0
        messages = [{"role": "system", "content": "system"}, *read_pair("read"), *read_pair("edit", "Edit")]
        engine._ask = lambda _state, batch: {
            f"call_{pair.call_id}": 0.8 for pair in batch
        } | {
            f"result_{pair.call_id}": 0.1 if pair.call_id == "read" else 0.9 for pair in batch
        }
        output = engine.compress(messages, force=True)
        self.assertIn("truncated", output[2]["content"])
        self.assertEqual(output[3]["tool_calls"][0]["id"], "edit")
        self.assertEqual(output[4]["tool_call_id"], "edit")

    def test_all_drop_fails_open(self):
        engine = Engine()
        engine.protect_first_n = 0
        engine.protect_last_n = 0
        messages = [{"role": "system", "content": "system"}, *read_pair("read")]
        engine._ask = lambda _state, batch: {
            f"{kind}_{pair.call_id}": 0.0 for pair in batch for kind in ("call", "result")
        }
        self.assertEqual(engine.compress(messages, force=True), messages)

    def test_register_and_deepcopy_match_the_hermes_plugin_contract(self):
        class PluginContext:
            engine = None

            def register_context_engine(self, engine):
                self.engine = engine

        context = PluginContext()
        ENGINE_MODULE.register(context)
        self.assertIsInstance(context.engine, Engine)
        clone = copy.deepcopy(context.engine)
        self.assertIsInstance(clone, Engine)
        self.assertIsNot(clone._lock, context.engine._lock)


if __name__ == "__main__":
    unittest.main()
