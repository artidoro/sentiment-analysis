# sentiment-analysis
Models for sentiment analysis including Naive Bayes, Logistic Regression, CBoW, CNN, 
LSTM, LSTM-CNN applied to SST2

<ul>
  <li> `main.py` is the file that calls the models</li>
  <li> `utils.py` contains useful functions to preprocess the dataset and generate iterators.</li> 
  <li> `model.py` is where all our models are implemented</li> 
  <li> `ensemble.py` allows to generate new predictions based on a majority vote of predictions from other models</li>
</ul>

We reached accuracy of 0.86352 on SST2.
https://www.kaggle.com/c/harvard-cs281-hw1/leaderboard

