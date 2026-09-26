# MagibuMizan

Typed probabilistic decisions from an open language model, without generating an answer string.

MagibuMizan presents a state and a typed question to a model, labels the possible answers `A`, `B`, `C`, …, and reads the next-token probability of each label. It reads every question with the options in normal and reversed order, averages the aligned probabilities, then optionally applies temperature scaling. The response contains a `choice`, a `score`, or a yes probability (`noul`).

The HTTP endpoint uses the request and answer fields of [TypeSafe's `/v1/systemone` API](https://docs.typesafe.ai/api) within the limits below. MagibuMizan is an independent open-model implementation, not a TypeSafe or Jev model.

## Install

Python 3.10 or newer is required. Install from PyPI:

```bash
pip install "magibumizan[mlx,server]"     # Apple Silicon
# or: pip install "magibumizan[cuda,server]" on an NVIDIA host
```

You can also install directly from this repository:

```bash
git clone https://github.com/magibu-ai/MagibuMizan.git
cd MagibuMizan
pip install ".[mlx,server]"     # Apple Silicon
# or: pip install ".[cuda,server]" on an NVIDIA host
```

The model weights are downloaded separately by the model library. The example [Gemma 4 MLX checkpoint](https://huggingface.co/mlx-community/gemma-4-26B-A4B-it-qat-4bit) is about 15.6 GB. Check the model's own license before using it.

## Python API

```python
from magibumizan import MagibuMizan

mizan = MagibuMizan(
    "mlx-community/gemma-4-26B-A4B-it-qat-4bit",
    backend="mlx",
    temperature=2.5,
)
answers, input_tokens = mizan.answer(
    "Siparişim 10 gündür gelmedi, kargo takip numarası da çalışmıyor. Paramı geri istiyorum.",
    {
        "iade": {"type": "noul", "instructions": "Müşteri para iadesi istiyor mu?"},
        "konu": {
            "type": "choice",
            "instructions": "Talebin konusu nedir?",
            "criteria": {
                "kargo": "Teslimat ve kargo",
                "iade": "İade ve para geri ödemesi",
                "urun": "Ürün kusuru",
                "diger": "Diğer",
            },
        },
        "ofke": {
            "type": "score",
            "instructions": "Müşterinin sinirlilik düzeyi?",
            "criteria": ["Sakin", "Tedirgin", "Sinirli", "Çok sinirli"],
        },
    },
)
print(answers)
```

One local M2 Pro run of this example used 690 input tokens and took about 1.6 seconds after loading the model. The output selected `iade`, estimated `noul=0.9974` for the refund request, and gave the frustration rubric a score of `2.444`. Results and latency depend on the model and hardware.

`noul` may include optional `criteria` with `true` and `false` descriptions. `choice` takes 2–26 keyed options. `score` takes 2–10 ordered levels. `state`, `instructions`, and descriptions can be strings or JSON objects/arrays. Returned `probabilities` are unrounded and sum to approximately 1; `confidence` and `score` are rounded to four decimals.

## HTTP API

```bash
MODEL=mlx-community/gemma-4-26B-A4B-it-qat-4bit \
TEMPERATURE=2.5 \
API_KEY=replace-with-a-secret \
uvicorn magibumizan.api:app --host 127.0.0.1 --port 8000
```

```bash
curl http://127.0.0.1:8000/v1/systemone \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer replace-with-a-secret' \
  -d '{"model":"jev-latest","state":"Hesabıma giremiyorum","questions":{"urgent":{"type":"noul","instructions":"Acil mi?"}}}'
```

`MODEL` selects the server's local model. A request's `model` field is accepted for client compatibility but does not change that selection. `API_KEY` is optional; set it before exposing the service. `BACKEND` overrides automatic selection (`mlx` on macOS, `cuda` elsewhere). `MAX_QUESTIONS` defaults to 10. `GET /health` reports readiness. MLX inference is processed serially in the server process.

The endpoint returns `401` for a missing or invalid API key, `422` for an invalid request, and `503` when local model inference fails. The server logs the underlying inference error.

## Calibration and evaluation

`temperature` must be positive. The default `1.0` leaves the averaged distribution unscaled. `2.5` above is an estimate fitted to a Turkish MMLU subset with this 4-bit Gemma checkpoint; it is **not** a universal confidence guarantee. Fit and check a temperature on held-out examples from your own task before using probabilities to automate consequential decisions.

See [BENCHMARKS.md](https://github.com/magibu-ai/MagibuMizan/blob/main/BENCHMARKS.md) for the evaluation method, model-specific accuracy, calibration results, dataset links, and limits of the comparisons.

## Limits

- The Turkish prompt and the listed temperature values were evaluated on Turkish tasks. Other languages and domains need their own evaluation.
- Choice supports at most 26 options because labels are single letters. The TypeSafe API allows more.
- Each question uses two model reads. The implementation does not share the state prefix between questions.
- The CUDA path is experimental until an end-to-end NVIDIA smoke test is recorded for this release.
- The package produces distributions over the supplied options; it cannot discover an omitted answer. Add an `other` or `none` option where appropriate.

## Development

```bash
pip install ".[server,test]"
python -m unittest discover -s tests -v
```

MIT licensed. MagibuMizan is not affiliated with TypeSafe AI.
