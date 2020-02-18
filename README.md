# Genre_Classification_Models

# Aim
For this project I wanted to build a multi-class classification model that could discrimate between multiple generes of music. I wanted to intentionally choose a handful of like genres which shared multiple features as well as distinct generes that shared few if any features. The purpose was to find a classification model that could be used to pinpoint differences between a genre's musical profile and help anaylize what features within the sound that made it unique. 

# Overview of Data
Using Spotify's API I called 14 different genres. These generes were: 

* 50’s (2)
* Chill Hop (1)
* Classical  (3)
* Detroit Techno (1)
* Disco (4)
* Electronic (4)
* French (5)
* Hip Hop (1)
* Industrial Pop (4)
* Post Rock (3)
* Rockabilly (2)
* Ska (2)
* Sleep (3)
* Spanish (5)

For each genre I called 100 songs totalling 1,400 tracks. Each track is given a number. Tracks with the same number share common audio features. 

# Spotify's Audio Features and Analysis

Spotify returns both high dimesional and low dimensional audio features. The high dimensional features are: 

* Valence - (mood)
* Time Signature 
* Tempo
* Speechiness 
* Loudness
* Liveness 
* Key - (0-11)
* Mode (-1, 0, 1)
* Instrumentalness
* Energy
* Danceability
* Acousticness

While somewhat inaccurate and in need of improvement, the lower dimensional features are: 

* Sections
  * Tempo of Sections 
  * Key of Sections (harmony) (0 - 11)
    * Segments
      * Pitch of Segment (note) - (0-11)
      * TImbre of Segment (color) - (0 - 1)

An example of what a genere's musical profile based off the high dimensional features is below. 
![50's Genre](https://github.com/FinchMF/Classification_Models/blob/master/graphs/EDA_50s_plots_1.png)

# Musical Profiles that share common features

Below is a graph showing three genres, Hip Hop, Chill Hop and Detroit Techno, that all share multiple common audio features. This particular graph is of the audio feature ENERGY and its distribution amoung the tracks within these three genres. 

![Energy Distro](https://github.com/FinchMF/Classification_Models/blob/master/graphs/3_togt.png)

# Model Overview

For a base model I used sklearn's Dummy Classifier, which had an Accuracy of 13% with each genre's precision at: 

* 50’s = precision: 0%
* Chill Hop = precision: 5%
* Classical = precision: 6%
* Detroit Techno = precision: 11%
* Disco = precision: 0%
* Electronic = precision: 0%
* French = precision: 0%
* Hip Hop = precision: 5%
* Industrial Pop = precision: 8%
* Post Rock = precision: 27%
* Rockabilly = precision: 0%
* Ska = precision: 12%
* Sleep = precision: 8%
* Spanish = precision: 25%

Here is the confusion matrix expressing the base model. 

![basemodel_14](https://github.com/FinchMF/Classification_Models/blob/master/graphs/base_model_15_cl.png)

You can see that a 15th classifier appears. This left me wondering if there was any noise in the machine making some tracks unclassifable and being automatically inserted into a default NAN category. More on this when we reach the final model.

Here are the results from each model I tried before concluding that Random Forest with a max depth of 10 was the most accurate model. 


* KNN: 
   * Accuracy: 30%

* Log Regression:
   * Accuracy: 43%

* Ada Boost:
  * Trained: 23.3%
  * Tested: 20.1%

* Gradient Boost: (overfitting)
  * Trained: 95.8%
  * Tested: 55.2%

* Random Forest:
  * Accuracy: 60.4%

* SVM:
  * Accuracy: 24.5%


On most models, the accuracy was too low and the recall/precision often had many classifiers scored at 0%. It was clear that these models were ineffecient and did not learn well. Intially, Random Forest did not reach 60.4%. It was only after tinkering with the max depth that the score jumped up. The final max depth was 10. 

![rndmfrstmxd10](https://github.com/FinchMF/Classification_Models/blob/master/graphs/forest_model_15.png)

Still the ghost classifier was appearing. After investigating further, I arrived at the conclusion that during the Test/Train, Split (which was an 80/20 divide) a NAN column was being generated to receive random tracks that were difficult for the machine to classify. This is something I want to look further into. 

Is it a flaw with Spotify's audio features, a flaw with Random Forest or a flaw with my code? In a future project, I will be sure to unpack the noise in this model. 

# Feature Importance for Random Forest at max depth 10 with 15 Classifiers (including NAN)

![feat_importance](https://github.com/FinchMF/Classification_Models/blob/master/graphs/feature_importance_15.png)


# Removing the Noise

Unsatisfied with being able to pinpoint what the noise was, I decided to strip the classifiers from the model that may be causing the noise. I did this by accessing the most distince genre's by looking at the ones that did not overlap that much. An example of instrumentalness between Classical and Electronic below. 

![difference](https://github.com/FinchMF/Classification_Models/blob/master/graphs/2_differ.png)

This lead me to choosing the following 6 genres: 

* 50’s
* Classical
* Electronic
* Industrial Pop
* Rockabilly
* Spanish

# Base Model for 6 Classifiers

![base_model_6](https://github.com/FinchMF/Classification_Models/blob/master/graphs/base_model_6_cl.png)

Notice that the noise is gone. 

The final model is Random Forest at max depth 10 with an accuracy of 80.4%

![random_forest-6](https://github.com/FinchMF/Classification_Models/blob/master/graphs/forest_model_6.png)

# Next Steps

 * Collect more data and further analyze where machine learning is less discriminate between like genres with lower dimensional features. 

* In the field of ethnomusicology, use the model to observe and isolate characteristics of a region’s or culture’s musical profile.


