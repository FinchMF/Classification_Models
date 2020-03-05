# Evaluating Classification Models

# Aim
Use data from Spotify's API as well as Engineered Features extrapolated from lower dimensional Spotify data to train and evaluate classification model performance. 

# Overview of Data
Using Spotify's API I called 14 different genre playlists. These generes were: 

* 50’s (2)
* Chill Hop (1)x
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




