from inference_sdk import InferenceHTTPClient

CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="9m8nciRcwcJFLJg8q1vl"
)
your_image = r"D:\Medias\fotograflar_Karasu_Belediyesi_Foto\fotograf_251034_20250828_061318.png"
result = CLIENT.infer(your_image, model_id="license-plate-recognition-rxg4e/11")