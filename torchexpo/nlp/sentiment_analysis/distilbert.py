import transformers
from torchexpo.modules import SentimentAnalysisModule


def distilbert_imdb():
    """DistilBERT Model pre-trained on IMDb"""
    model = transformers.DistilBertForSequenceClassification.from_pretrained(
        "textattack/distilbert-base-uncased-imdb", torchscript=True)
    obj = SentimentAnalysisModule(model, "DistilBERT IMDb", model_example="default")
    return obj