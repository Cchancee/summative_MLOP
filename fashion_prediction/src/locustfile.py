from locust import HttpUser, task, between

class PredictUser(HttpUser):
    wait_time = between(1, 3)  # Simulates user wait between requests

    @task
    def predict_image(self):
        with open("sample-image.png", "rb") as f:
            files = {"file": ("sample-image.png", f, "image/png")}
            self.client.post("/predict", files=files)
