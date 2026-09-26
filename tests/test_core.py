import unittest
from unittest.mock import patch

from magibumizan import MagibuMizan


class FakeLM:
    rows = []

    def __init__(self, _name):
        self.tok = self.chat = self
        self.prompts = []

    def encode(self, text, add_special_tokens=False):
        if len(text) == 1 and text.isupper():
            return [ord(text)]
        if len(text) == 2 and text[0] == " " and text[1].isupper():
            return [32, ord(text[1])]
        return list(text.encode("utf-8"))

    def apply_chat_template(self, messages, **_kwargs):
        return messages[0]["content"]

    def probs(self, prompts, ids):
        self.prompts = prompts
        return [[row[identifier - ord("A")] for identifier in ids] for row in self.rows]


def padded(*values):
    return list(values) + [0.0] * (26 - len(values))


class CoreTests(unittest.TestCase):
    def make_engine(self, rows, temperature=1):
        FakeLM.rows = rows
        with patch("magibumizan.core.MLX", FakeLM):
            return MagibuMizan("fake/model", backend="mlx", temperature=temperature)

    def test_all_question_types_and_reversed_options(self):
        engine = self.make_engine([
            padded(0.8, 0.2), padded(0.3, 0.7),
            padded(0.1, 0.7, 0.2), padded(0.6, 0.3, 0.1),
            padded(0.1, 0.2, 0.7), padded(0.3, 0.6, 0.1),
        ])
        answers, tokens = engine.answer("A case", {
            "yes": {"type": "noul", "instructions": "Is it true?",
                    "criteria": {"true": "Explicit evidence", "false": "No evidence"}},
            "kind": {"type": "choice", "instructions": "Which kind?",
                     "criteria": {"one": None, "two": "Second", "three": "Third"}},
            "level": {"type": "score", "instructions": "How much?",
                      "criteria": ["Low", "Mid", "High"]},
        })
        self.assertEqual(answers["yes"]["noul"], 0.75)
        self.assertEqual(answers["kind"]["choice"], "two")
        self.assertAlmostEqual(answers["kind"]["probabilities"]["two"], 0.5)
        self.assertAlmostEqual(answers["level"]["score"], 1.4)
        self.assertGreater(tokens, 0)
        self.assertIn("A: Evet (Explicit evidence)", engine.lm.prompts[0])
        self.assertIn("A: Hayır (No evidence)", engine.lm.prompts[1])
        self.assertIn("A: three (Third)", engine.lm.prompts[3])
        for key in ("kind", "level"):
            self.assertAlmostEqual(sum(answers[key]["probabilities"].values()), 1.0)

    def test_temperature_changes_distribution(self):
        engine = self.make_engine([padded(0.9, 0.1), padded(0.1, 0.9)], temperature=2)
        answer, _ = engine.answer("State", {
            "x": {"type": "noul", "instructions": "Question"}
        })
        self.assertEqual(answer["x"]["noul"], 0.75)

    def test_invalid_requests_fail_before_inference(self):
        engine = self.make_engine([])
        bad_questions = [
            {},
            {"x": {"type": "choice", "instructions": "x", "criteria": {"only": None}}},
            {"x": {"type": "score", "instructions": "x", "criteria": list(range(11))}},
            {"x": {"type": "noul", "instructions": "x", "criteria": {"other": "x"}}},
            {"x": {"type": "noul"}},
        ]
        for questions in bad_questions:
            with self.subTest(questions=questions), self.assertRaises(ValueError):
                engine.answer("State", questions)
        self.assertEqual(engine.lm.prompts, [])

    def test_invalid_configuration(self):
        for temperature in (0, -1, float("nan"), float("inf")):
            with self.subTest(temperature=temperature), self.assertRaises(ValueError):
                MagibuMizan("fake/model", backend="mlx", temperature=temperature)
        with self.assertRaises(ValueError):
            MagibuMizan("fake/model", backend="other")


if __name__ == "__main__":
    unittest.main()
