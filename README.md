# Evaluating Classification Models

# Aim
Use data from Spotify's API as well as Engineered Features extrapolated from lower dimensional Spotify data to train and evaluate classification model performance. 

# Overview of Data
Using Spotify's API I called 14 different genre playlists. These generes were: 

* 50’s (2)
* Classical  (3)
* Detroit Techno (1)
* Disco (4)
* Electronic (4)
* Electro Indie Pop (4)
* French (5)
* Hip Hop (1)
* Industrial Pop (4)
* Post Rock (3)
* Rockabilly (2)
* Ska (2)
* Sleep (3)
* Spanish (5)

For each genre 100 tracks, totalling 1,400, were called. Each genre is given a number. Genres with the same number share common audio features. 

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

 The lower dimensional features are: 

* Sections
  * Tempo of Sections 
  * Key of Sections (harmony) (0 - 11)
    * Segments
      * Pitch of Segment (note) - (0-11)
      * TImbre of Segment (color) - (0 - 1)


## Musical Profile of Genre      

Three examples of a genere's musical profile based off the high dimensional features are below. These examples are chosen because they share similarities and express overt differences. 


### 50's Musical Profile

![50's Genre](/Volumes/S190813/Coding/flatiron/Classification_Models/Classification_Models/graphs/EDA_50s_plots_1.png)

### French Musical Profile

![French Genre](/Volumes/S190813/Coding/flatiron/Classification_Models/Classification_Models/graphs/EDA_french_plots_2.png)

### Hip Hop Profile

![Hip_Hop Genre](/Volumes/S190813/Coding/flatiron/Classification_Models/Classification_Models/graphs/EDA_hip_hop_plots_3.png)


Between these examples you can see that there is one feature that is common between all of them. The common feature is categorical: Instrumentalness. 

But between features, like Duration and Energy, the genre's differ in their continuous values. The Musical Profile of a genre is a distilled portrait of the sonic spectrum a genre exists within. 


# Musical Profiles that share common features

Below is a graph showing three genres, Hip Hop, Chill Hop and Detroit Techno, that all share multiple common audio features. This particular graph is of the audio feature ENERGY. In the common feature space, the question is what allows each genre to be 

![Energy Distro](/Volumes/S190813/Coding/flatiron/Classification_Models/Classification_Models/graphs/3_togt.png)

# Model Evaluations

   
Using sklearn's Dummy Classifier, a base model with an Accuracy of 6.7% was achieved with percision scores for each genre below: 

* 50’s = precision: 9%
* Classical = precision: 0%
* Detroit Techno = precision: 0%
* Disco = precision: 38%
* Electronic = precision: 0%
* Electro Indie Pop = precision: 5%
* French = precision: 0%
* Hip Hop = precision: 3%
* Industrial Pop = precision: 14%
* Post Rock = precision: 0%
* Rockabilly = precision: 0%
* Ska = precision: 15%
* Sleep = precision: 18%
* Spanish = precision: 0%

Here is the confusion matrix expressing the base model. 

![basemodel_14](/Volumes/S190813/Coding/flatiron/Classification_Models/Classification_Models/graphs/base_model_15.png)


In order to find a model and optimize it, I wrote a script detailing two main functions. Find_Model and Run_Model. 

Find_Model takes in a dataframe, a set of features, the model type and number of classifiers respective to the size of the dataframe. The function uses train/test split and targets genre classifiers with the chosen features. It then executes a base model then uses gridsearch to find an optimized model. The function saves both the non optimzed and optimized model. 

Run_Model takes in a dataframe, a set of features, loads a model and a number of classifiers respective to the dataframe and outputs the model's accuracy score, a classification report, confusion matrix and a list of the model's classification results. 

### An Example of Evaluating a Support Vector Classifier with Find_Model function. 

The intial output of Linear SVC with 14 classifiers. 
![SVC](/Volumes/S190813/Coding/flatiron/Classification_Models/Classification_Models/graphs/SVC_14_non_optimized.png)

The accuracy score @ 0.3319327731092437 is better than the Base Model but still a poor classifier.

Below is a confusion matrix after the GridSearch: 

![SVC_optimized](/Volumes/S190813/Coding/flatiron/Classification_Models/Classification_Models/graphs/SVC_14_optimized.png)

The accuracy score @ 0.5714285714285714


In order to further evaluate the model other than searching for optimal parameters, lets look at the classifiers and see where the model often misclassified. In the print statement of Find_Model, as well as Run_Model functions, you will notice a list of tuples that show the predicted genre and actual genre. In addition to looking at what genres the model misclassified the most, we can also look at what combination of features are effective in training the model and which lessen the trainability. 

### Engineered Features

For this project, there were three features I engineered. They are abstractions of Spotify's data and an attempt to further depict the form of a song. Using data from 'Audio Features' and 'Audio Analysis' endpoints, I was able to initially generate three depictions of a track's form. 

1. Harmonic Progression
2. Modal Progression
3. Tempo Progression

Combining both Harmonic and Modal Progression, I was able to generate categorical features that depicted a Harmonic Signature of a peice. A signature expresses the general shape of harmonic modality as well as isolates unique chords. For this project, I did not make use of the Tempo Progression feature. 


