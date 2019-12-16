# Genre_Classification_Models

# Aim
For this project I wanted to build a multi-class classification model that could discrimate between multiple generes of music. I wanted to intentionally choose a handful of like genres which shared multiple features as well as distinct generes that shared few if any features. The purpose was to find a classification model that could be used to pinpoint differences between a genre's musical profile and help anaylize what features within the sound that made it unique. 

# Overview
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

Three examples  of what a genere's musical profile based off the high dimensional features is below. 
![50's Genre](https://github.com/FinchMF/Classification_Models/blob/master/graphs/EDA_50s_plots_1.png)
![French Genre](https://github.com/FinchMF/Classification_Models/blob/master/graphs/EDA_french_plots_2.png)
![Hip_hop](https://github.com/FinchMF/Classification_Models/blob/master/graphs/EDA_hip_hop_plots_3.png)
