# Predicting Polarization From News Outlets Tweets

**Team Members**  
* Stephanie Ramos Gomez
* Kelsey Anderson
* Jesica Ramirez Toscano

## Table of Contents 

* [Summary](#summary)
* [Set up](#set-up)
* [Data Collection and Pre-processing](#data-collection-and-pre-processing)
* [Data Exploration](#data-exploration)
* [Model Building](#model-building)
   * bi-RNN
   * CNN
   * Logistic Regression
* [Model Interpretability: Integrated gradients](#model-interpretability)
* [Final Paper and Acknowledgements](#final-paper-and-acknowledgements)

## Summary 
Media bias has been a topic of interest for policy makers and politicians particu- larly since the 2020 US presidential election. Society appears to be increasingly polarized and hostile in online interactions. Online spaces have become the target of criticism and fears about cyberbullying, trolling, fake news and the potential for vulnerable individuals to become violently radicalized. With this in mind, we built different supervised machine learning models to test if the text extracted from tweets posted by news media accounts is predictive of the polarization in the comments they receive. We used logistic regression, recurrent neural networks and convolutional neural networks to make our predictions. Using these models and data extracted from Twitter, we were able to predict polarization in comments with 65% accuracy. Our research aims to contribute to creating a more civil and less polarized space for discourse on social media.

## Set Up
The following modules were used for this analysis:
```
torch
nltk
gensim
sklearn
vaderSentiment
networkx
pandas
numpy
matplotlib
seaborn
snscraper
camptum
```
### Tweet Analysis Pipeline
To pull, clean and run models using Twitter data scraped with [**snscraper**](https://github.com/JustAnotherArchivist/snscrape), run, in order:
1. tweet_scraping.ipynb
2. Clean tweets.ipynb
3. topics_twitter.ipynb
4. sentiment_scores.ipynb
5. Model of Choice:
    - Log Reg.ipynb
    - rnn_model.ipynb
    - cnn_development.ipynb

### Kaggle NYT Article Pipeline
To pull, clean and run models using downloaded Kaggle NYT articles data, run, in order:
1. download data from [New York Times Comments on Kaggle](https://www.kaggle.com/aashita/nyt-comments)
2. Data Cleaning.ipynb
3. topics_articles.ipynb
4. sentiment_scores.ipynb
5. Model of Choice:
    - Log Reg.ipynb
    - rnn_model.ipynb
    - cnn_development.ipynb

## Data Collection and Pre-processing
The following files are used in data collection and text preprocessing.

### Tweet Scraping
#### `tweet_scraping.ipynb`  
*lines of code: 850*
  
Uses  [**snscraper**](https://github.com/JustAnotherArchivist/snscrape) to retrieve original posts from NYT, FoxNews and Reuters Twitter feeds. The file name and date ranges for the search can be specified in the notebook parameters. After gathering original tweets, it attempts to retrieve all subsequent replies based on the ConversationID of the original tweet.

Saves one file for all original tweets: `data/tweets_MMYYYY.csv` and a separate file for each news source's replies: `data/replies_MMYYYY_nyt.csv`, `data/replies_MMYYYY_fox.csv` and `data/replies_MMYYYY_reu.csv`. 

**Warning**: This file can take nearly 12 hours to complete a datapull for a single month span of time.


### Data Cleaning
#### `Clean tweets.ipynb`
*lines of code: 564*

#### `Data Cleaning.ipynb`
*lines of code: 93*  
Same as Clean tweets.ipynb, but built to accept Kaggle NYT data.

### Creating Target Column: Sentiment Scores
#### `sentiment_scores.ipynb`  
*lines of code: 677*

Combines replies into a measure of setiment variance, merges with the original post dataset and outputs for use in our machine learning models.
Can accept either the Kaggle NYT dataset or a Twitter dataset by commented/uncommenting parameters at the beginning of the notebook.

We attempted numerous ways to bin the sentiment scoring data. Current state is using a simplified count method. Please read our [**project paper**](documents/Predicting_Polarization.pdf) for more detials on our methodolgy and experiements with this.

This notebook uses [**VADER**](https://github.com/cjhutto/vaderSentiment#installation) to determine and score positive, negative and neutral tweet sentiment.

**Warning:** If the `replies_SUFFIX.csv` is large, evaluating sentiment can take a considerable time period. Average time for our full dataset is approximately 20-30 minutes.



## Data Exploration
The following files were used in data exploration.
Topic modeling files can be included in the data cleaning pipeline, or skipped, as desired.
It was originally intended as a way to filter data fed into the models, however, due to lack of data quantity, it is not currently so used by any of our models.

### Topic Modeling
#### `topics_twitter.ipynb`
*lines of code: 1451*

#### `topics_articles.ipynb`
*lines of code: 1980*  
Same as topics_twitter.ipynb, but built to accept Kaggle NYT data.


### Examing an Alternative Measure of Polarity
#### `Controversy_scores.ipynb`
*lines of code: 527*

## Model Building

### Bi-Directional Recurrent Neural Network
#### `rnn_model.ipynb` 
*lines of code: 171+*

Trains and tests an LSTM RNN model on the desired dataset.
Datafile can be specified in the parameters of the notebook, but requires a cleaned text feature column named `text` and a binary target column called `vaderCat`.
If your target column is labelled differently, this will need to be adjusted inside of the `train_test_datasets.py` sub-module.


#### Also includes Integrated Gradient Model Interprebility Code  
*lines of code: 81 of total 171*  

This section is found near the end of the notebook. It produces an estimation of feature importance over the test dataset after the model is trained.

##### Uses sub-modules:
1. **`train_test_datasets.py`**  
    *lines of code: 120*  
    
    Imports and splits the full dataset into train.csv, validate.csv and test.csv to be received by the dataloaders internal to **rnn_model.ipynb**.
    Based on CAPP 30255 Homework 4.
    
    
2. **`lstm.py`**  
   *lines of code: 53*  
   
    Contains the PyTorch RNN model object specifications.
    This is modularized so other model specifications could be tested. The best performing specifications are saved here.
    Based on [**Fake News Detection**](https://github.com/bentrevett/pytorch-sentiment-analysis/blob/master/1%20-%20Simple%20Sentiment%20Analysis.ipynb).
    
 3. **`evaluate.py`**  
   *lines of code: 199*  

   Contains training and test evaluation functions used in `rnn_model.ipynb`. 
   Based on [**Fake News Detection**](https://github.com/bentrevett/pytorch-sentiment-analysis/blob/master/1%20-%20Simple%20Sentiment%20Analysis.ipynb).
   

### Convolutional Neural Network
#### `cnn_development.ipynb`
*lines of code: 412 + 109 (IG section)*
##### Uses sub-modules:
1. **`dataclassCNN.py`** 
    *lines of code: 35* 

    File to create a Custom Data Class and Collate Function for PyTorch.


### Logistic Regression
#### `Log Reg.ipynb`
*lines of code: 927*

## Model Interpretability 
### Feature importances using coefficients
To get a sense of the feature importance in the ligistic regression model we ordered the coefficients by magnitude.
The code is at the end of the ligistic regression notebook:
* Log Reg.ipynb

### Integrated Gradients 

Integrated Gradient is an interpretability algorithm used to understand how neural networks work. This algorithm, proposed by Sundararajan et al. (2017), attributes the prediction of a neural network to its input features. 

The code to apply integrated gradients/feature importannce) is at the end of the notebook for each model:
* rnn_model.ipynb
* cnn_development.ipynb



## Final Paper and Acknowledgements
The research, analysis and results of this project are documented in [**Predicting_Polarization.pdf**](documents/Predicting_Polarization.pdf).
We want to thank professor **Amitabh Chaudhary** for his incredible support and feedback.
Also, huge thanks to the Teaching Assistants: Jenny Long and Rosa Zhou.
