from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import softmax
import torch

class EmotionDetector:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("nateraw/bert-base-uncased-emotion")
        self.model = AutoModelForSequenceClassification.from_pretrained("nateraw/bert-base-uncased-emotion")
        self.labels = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']

    def predict(self, text):
        inputs = self.tokenizer(text, return_tensors="pt")
        outputs = self.model(**inputs)
        probs = softmax(outputs.logits, dim=1)
        confidence, prediction = torch.max(probs, dim=1)
        return self.labels[prediction.item()], confidence.item()
