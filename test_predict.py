import unittest

from predict import app


class PredictionApiTest(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_health(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), {"status": "ok"})

    def test_rejects_invalid_payload(self):
        response = self.client.post("/api/predict", json={})
        self.assertEqual(response.status_code, 400)

    def test_predicts_supported_label(self):
        response = self.client.post("/api/predict", json={"text": "You are an idiot"})
        self.assertEqual(response.status_code, 200)
        self.assertIn(response.get_json()["prediction"], {"Hateful Content", "Offensive Content"})

    def test_low_confidence_benign_message_is_not_flagged(self):
        response = self.client.post(
            "/api/predict",
            json={"text": "[UX TEST] Real-time message from Maurya"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["prediction"], "Neither")


if __name__ == "__main__":
    unittest.main()
