import transformers
from torchexpo.modules import SentimentAnalysisModule


def electra_imdb():
    """Electra Model"""
    model = transformers.ElectraForSequenceClassification.from_pretrained(
        "monologg/electra-small-finetuned-imdb", torchscript=True)
    obj = SentimentAnalysisModule(model, "Electra IMDb")
    return obj