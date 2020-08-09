import transformers
from torchexpo.modules import SentimentAnalysisModule


def distilbert_imdb():
    """DistilBERT Model"""
    model = transformers.DistilBertForSequenceClassification.from_pretrained(
        "textattack/distilbert-base-uncased-imdb", torchscript=True)
    obj = SentimentAnalysisModule(model, "DistilBERT IMDb")
    return obj