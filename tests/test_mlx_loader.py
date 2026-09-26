import sys
import types
import unittest
from unittest.mock import Mock, patch

from magibumizan.core import MLX


class MLXLoaderTests(unittest.TestCase):
    def load_with(self, vlm_load, lm_load):
        modules = {}
        for package in ("mlx", "mlx_vlm", "mlx_vlm.models", "mlx_lm", "mlx_lm.models"):
            modules[package] = types.ModuleType(package)
            modules[package].__path__ = []
        modules["mlx.core"] = types.ModuleType("mlx.core")
        modules["mlx_vlm.models.cache"] = types.ModuleType("mlx_vlm.models.cache")
        modules["mlx_lm.models.cache"] = types.ModuleType("mlx_lm.models.cache")
        modules["mlx_vlm"].load = vlm_load
        modules["mlx_lm"].load = lm_load
        modules["mlx_vlm.models.cache"].make_prompt_cache = lambda model: ("vlm", model)
        modules["mlx_lm.models.cache"].make_prompt_cache = lambda model: ("lm", model)
        with patch.dict(sys.modules, modules):
            return MLX("example/model")

    def test_multimodal_checkpoint_uses_vlm_language_model(self):
        language_model = object()
        tokenizer = types.SimpleNamespace(chat_template="template")
        processor = types.SimpleNamespace(tokenizer=tokenizer)
        vlm_load = Mock(return_value=(types.SimpleNamespace(language_model=language_model), processor))
        lm_load = Mock()
        backend = self.load_with(vlm_load, lm_load)
        self.assertIs(backend.model, language_model)
        self.assertIs(backend.tok, tokenizer)
        self.assertIs(backend.chat, tokenizer)
        self.assertEqual(backend.make_cache(language_model), ("vlm", language_model))
        lm_load.assert_not_called()

    def test_unsupported_vlm_type_uses_lm(self):
        model, tokenizer = object(), object()
        vlm_load = Mock(side_effect=ValueError("Model type llama not supported. Error: missing module"))
        lm_load = Mock(return_value=(model, tokenizer))
        backend = self.load_with(vlm_load, lm_load)
        self.assertIs(backend.model, model)
        self.assertIs(backend.tok, tokenizer)
        self.assertEqual(backend.make_cache(model), ("lm", model))
        lm_load.assert_called_once_with("example/model")

    def test_vlm_load_failure_is_not_hidden_by_lm(self):
        lm_load = Mock()
        with self.assertRaisesRegex(ValueError, "missing weights"):
            self.load_with(Mock(side_effect=ValueError("missing weights")), lm_load)
        lm_load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
