import unittest

from fastapi.testclient import TestClient

from magibumizan.api import create_app
from magibumizan.core import RequestValidationError


class FakeEngine:
    def answer(self, state, questions):
        if not isinstance(state, str) or not isinstance(questions, dict):
            raise ValueError("invalid input")
        return {"q": {"type": "noul", "noul": 0.75}}, 42


class FailingEngine:
    def __init__(self, error):
        self.error = error

    def answer(self, state, questions):
        raise self.error


class ApiTests(unittest.TestCase):
    def test_response_and_authentication(self):
        app = create_app(FakeEngine(), model="fake/model", api_key="secret")
        with TestClient(app) as client:
            body = {"state": "A case", "model": "jev-latest", "questions": {
                "q": {"type": "noul", "instructions": "Is it true?"}
            }}
            self.assertEqual(client.post("/v1/systemone", json=body).status_code, 401)
            response = client.post("/v1/systemone", json=body,
                                   headers={"Authorization": "Bearer secret"})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json(), {
                "model": "fake/model", "answers": {"q": {"type": "noul", "noul": 0.75}},
                "usage": {"input_tokens": 42, "output_tokens": 0},
            })

    def test_bad_body_and_question_limit(self):
        app = create_app(FakeEngine(), model="fake/model", max_questions=1)
        with TestClient(app) as client:
            self.assertEqual(client.post("/v1/systemone", json=[]).status_code, 422)
            self.assertEqual(client.post("/v1/systemone", json={"state": "x", "questions": {}}).status_code, 422)
            self.assertEqual(client.post("/v1/systemone", json={
                "state": "x", "questions": {"a": {}, "b": {}}
            }).status_code, 422)

    def test_non_ascii_api_key_and_inference_errors(self):
        body = {"state": "x", "questions": {"q": {"type": "noul", "instructions": "x"}}}
        app = create_app(FakeEngine(), model="fake/model", api_key="şifre")
        with TestClient(app) as client:
            self.assertEqual(client.post("/v1/systemone", json=body).status_code, 401)
        app = create_app(FailingEngine(RequestValidationError("bad criteria")), model="fake/model")
        with TestClient(app) as client:
            self.assertEqual(client.post("/v1/systemone", json=body).status_code, 422)
        for failure in (RuntimeError("model failed"), ValueError("bad model weights")):
            with self.subTest(failure=failure):
                app = create_app(FailingEngine(failure), model="fake/model")
                with TestClient(app) as client:
                    with self.assertLogs("magibumizan.api", level="ERROR"):
                        response = client.post("/v1/systemone", json=body)
                    self.assertEqual(response.status_code, 503)
                    self.assertEqual(response.json()["message"], "model inference failed")


if __name__ == "__main__":
    unittest.main()
