torchexpo.nlp
#############

.. automodule:: torchexpo.nlp
  :members:

Sentiment Analysis
==================

.. image:: https://res.cloudinary.com/torchexpo/image/upload/v1596868483/tasks/sentiment-analysis.jpg
   :width: 25%
   :align: right

|

Sentiment analysis is the task of classifying the polarity of a given text.

Example:
    >>> from torchexpo.nlp import sentiment_analysis
    >>> 
    >>> model = sentiment_analysis.electra_imdb()
    >>> model.extract_torchscript()
    >>> model.extract_onnx()

.. automodule:: torchexpo.nlp.sentiment_analysis
  :members:

DistilBERT (IMDb)
-----------------

.. autofunction:: torchexpo.nlp.sentiment_analysis.distilbert_imdb()


ELECTRA (IMDb)
--------------

.. autofunction:: torchexpo.nlp.sentiment_analysis.electra_imdb()

Text Classification (Coming Soon)
=================================

**Coming Soon**

Question Answering (Coming Soon)
================================

**Coming Soon**
