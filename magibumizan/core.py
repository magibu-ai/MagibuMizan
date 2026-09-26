"""Read Choice, Score, and Noul probabilities without generating text."""

import json
import math
import sys

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
PROMPT = (
    "Aşağıdaki duruma göre soruyu cevapla. Sadece doğru seçeneğin harfini yaz, açıklama yapma.\n\n"
    "Durum:\n{state}\n\nSoru: {question}\nSeçenekler:\n{options}"
)


class MLX:
    def __init__(self, name):
        import mlx.core as mx

        try:
            from mlx_lm import load
            from mlx_lm.models.cache import make_prompt_cache

            self.model, self.tok = load(name)
            self.chat = self.tok
        except Exception:  # Multimodal checkpoints such as Gemma 4 need mlx-vlm.
            from mlx_vlm import load
            from mlx_vlm.models.cache import make_prompt_cache

            model, processor = load(name)
            self.model, self.tok = model.language_model, processor.tokenizer
            self.chat = self.tok if getattr(self.tok, "chat_template", None) else processor
        self.mx, self.make_cache = mx, make_prompt_cache

    def probs(self, prompts, ids):
        mx, result = self.mx, []
        for prompt in prompts:
            tokens = self.tok.encode(prompt, add_special_tokens=False)
            if not tokens:
                raise RuntimeError("the chat template produced an empty prompt")
            cache = self.make_cache(self.model)
            for i in range(0, len(tokens) - 1, 512):
                self.model(mx.array(tokens[i : min(i + 512, len(tokens) - 1)])[None], cache=cache)
                mx.eval([entry.state for entry in cache])
            logits = self.model(mx.array(tokens[-1:])[None], cache=cache)
            logits = getattr(logits, "logits", logits)[0, -1].astype(mx.float32)
            result.append(mx.softmax(logits)[mx.array(ids)].tolist())
        return result


class CUDA:
    def __init__(self, name):
        import torch
        import transformers as tf

        self.torch = torch
        self.tok = tf.AutoTokenizer.from_pretrained(name)
        self.tok.pad_token = self.tok.pad_token or self.tok.eos_token
        self.tok.padding_side = "left"
        self.chat = self.tok
        for auto in ("AutoModelForMultimodalLM", "AutoModelForImageTextToText", "AutoModelForCausalLM"):
            try:
                self.model = getattr(tf, auto).from_pretrained(
                    name, dtype=torch.bfloat16, device_map="cuda"
                ).eval()
                break
            except (AttributeError, ValueError, KeyError, ImportError):
                continue
        else:
            raise ValueError(f"cannot load {name} with transformers")

    def probs(self, prompts, ids):
        encoded = self.tok(
            prompts, return_tensors="pt", padding=True, add_special_tokens=False
        ).to(self.model.device)
        with self.torch.inference_mode():
            logits = self.model(**encoded, logits_to_keep=1).logits[:, -1].float()
        return self.torch.softmax(logits, -1)[:, ids].tolist()


def _display(value):
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def _description(value):
    return "" if value is None else _display(value)


def _check_content(value, field):
    if not isinstance(value, (str, dict, list)):
        raise ValueError(f"{field} must be a string, object, or array")
    try:
        json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be JSON serializable") from exc


def _options(question):
    if not isinstance(question, dict):
        raise ValueError("each question must be an object")
    kind = question.get("type")
    if kind not in ("choice", "score", "noul"):
        raise ValueError(f"unknown question type {kind!r}")
    if "instructions" not in question:
        raise ValueError("question instructions are required")
    _check_content(question["instructions"], "instructions")

    criteria = question.get("criteria")
    if kind == "noul":
        if criteria is not None:
            if not isinstance(criteria, dict) or set(criteria) - {"true", "false"}:
                raise ValueError("noul criteria must contain only true and false descriptions")
            for key, value in criteria.items():
                _check_content(value, f"noul criteria.{key}")
        criteria = criteria or {}
        return [("Evet", _description(criteria.get("true"))),
                ("Hayır", _description(criteria.get("false")))]

    if kind == "choice":
        if not isinstance(criteria, dict) or not 2 <= len(criteria) <= len(LETTERS):
            raise ValueError("choice criteria must have 2–26 options")
        options = []
        for key, value in criteria.items():
            if not isinstance(key, str) or not key:
                raise ValueError("choice option keys must be nonempty strings")
            if value is not None:
                _check_content(value, f"choice criteria.{key}")
            options.append((key, _description(value)))
        return options

    if not isinstance(criteria, list) or not 2 <= len(criteria) <= 10:
        raise ValueError("score criteria must have 2–10 levels")
    for index, value in enumerate(criteria):
        _check_content(value, f"score criteria[{index}]")
    return [(_display(value), "") for value in criteria]


class MagibuMizan:
    """Answer typed questions with an MLX or CUDA language model."""

    def __init__(self, model, backend=None, temperature=1.0):
        if not isinstance(model, str) or not model:
            raise ValueError("model must be a nonempty string")
        if not isinstance(temperature, (int, float)) or not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be a finite number greater than zero")
        backend = backend or ("mlx" if sys.platform == "darwin" else "cuda")
        if backend not in ("mlx", "cuda"):
            raise ValueError("backend must be 'mlx' or 'cuda'")
        self.lm = (MLX if backend == "mlx" else CUDA)(model)
        self.temperature = float(temperature)
        variants = [
            {tokens[0] for tokens in (
                self.lm.tok.encode(form, add_special_tokens=False)
                for form in (letter, " " + letter)
            ) if len(tokens) == 1}
            for letter in LETTERS
        ]
        self.available = [bool(values) for values in variants]
        self.ids = [token for values in variants for token in sorted(values)]
        self.letter = [index for index, values in enumerate(variants) for _ in values]

    def _prompt(self, state, question, options):
        lines = "\n".join(
            f"{LETTERS[index]}: {key}" + (f" ({description})" if description else "")
            for index, (key, description) in enumerate(options)
        )
        content = PROMPT.format(state=_display(state), question=_display(question), options=lines)
        return self.lm.chat.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    def _distribution(self, raw, count):
        values = [0.0] * count
        for index, probability in zip(self.letter, raw):
            if index < count:
                values[index] += probability
        total = sum(values)
        if total <= 0 or not math.isfinite(total):
            raise RuntimeError("the model assigned no probability to the option letters")
        return [value / total for value in values]

    def _calibrate(self, forward, reverse):
        averaged = [(a + b) / 2 for a, b in zip(forward, reverse[::-1])]
        logits = [math.log(max(value, 1e-12)) / self.temperature for value in averaged]
        offset = max(logits)
        weights = [math.exp(value - offset) for value in logits]
        total = sum(weights)
        return [value / total for value in weights]

    def answer(self, state, questions):
        """Return ``(answers, input_tokens)`` for a map of typed questions."""
        _check_content(state, "state")
        if not isinstance(questions, dict) or not questions:
            raise ValueError("questions must be a nonempty object")
        checked = []
        for name, question in questions.items():
            if not isinstance(name, str) or not name:
                raise ValueError("question ids must be nonempty strings")
            options = _options(question)
            if not all(self.available[: len(options)]):
                raise ValueError("this model does not encode every required option letter as one token")
            checked.append((name, question, options))

        prompts = [
            self._prompt(state, question["instructions"], options[::direction])
            for _, question, options in checked
            for direction in (1, -1)
        ]
        raw = self.lm.probs(prompts, self.ids)
        answers = {}
        for index, (name, question, options) in enumerate(checked):
            count = len(options)
            probabilities = self._calibrate(
                self._distribution(raw[2 * index], count),
                self._distribution(raw[2 * index + 1], count),
            )
            confidence = round((count * max(probabilities) - 1) / (count - 1), 4)
            kind = question["type"]
            if kind == "noul":
                answers[name] = {"type": "noul", "noul": round(probabilities[0], 4)}
            elif kind == "choice":
                answers[name] = {
                    "type": "choice",
                    "choice": options[probabilities.index(max(probabilities))][0],
                    "confidence": confidence,
                    "probabilities": {
                        option[0]: probability for option, probability in zip(options, probabilities)
                    },
                }
            else:
                answers[name] = {
                    "type": "score",
                    "score": round(sum(i * probability for i, probability in enumerate(probabilities)), 4),
                    "confidence": confidence,
                    "probabilities": {str(i): probability for i, probability in enumerate(probabilities)},
                    "legend": {str(i): option[0] for i, option in enumerate(options)},
                }
        tokens = sum(len(self.lm.tok.encode(prompt, add_special_tokens=False)) for prompt in prompts)
        return answers, tokens
