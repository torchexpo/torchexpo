from torchexpo.nlp import sentiment_analysis


def test_electra_imdb():
    """Test Electra IMDb"""
    electra_imdb = [sentiment_analysis.electra_imdb()]
    map(extract_sentiment_analysis, electra_imdb)

def test_distilbert_imdb():
    """Test DistilBERT IMDb"""
    distilbert_imdb = [sentiment_analysis.distilbert_imdb()]
    map(extract_sentiment_analysis, distilbert_imdb)

def extract_sentiment_analysis(model):
    """Runs extraction common for all sentiment analysis models"""
    model.extract_torchscript()
    model.extract_onnx()