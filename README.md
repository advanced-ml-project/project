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
```
## Data Collection and Pre-processing

## Data Exploration

## Model Building

### Bi-Directional Recurrent Neural Network 
### Convolutional Neural Network
### Logistic Regression

## Model Interpretability 

## Final Paper and Acknowledgements
The research, analysis and results of this project are documented in [**Predicting_Polarization.pdf**](Predicting_Polarization.pdf).
We want to thank professor **Amitabh Chaudhary** for his incredible support and feedback.
Also, huge thanks to the Teaching Assistants: Jenny Long and Rosa Zhou.
