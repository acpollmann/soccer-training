---
htmlwidgets: true
---

#### **Team Members: Felipe Godoy, Andrew Gauker, Amy Pollman, Hetu Patel, Nouzhan Vakili Dastjerd**

# **Introduction**

### Background

Throughout history, sports have played an integral part in societies worldwide. In contemporary times, soccer reigns supreme as the world's most popular sport. Consequently, billions of fans all over the globe know dozens of teams, many players, and all their individual playing statistics. In a world where we are growing increasingly more divided, soccer remains one of the few pastimes that unites people across all boundaries. With over 3.6 billion viewers and half the world's population tuned into the 2018 World Cup, soccer's global importance is undeniable.

### What's the Problem?

Despite many fans having an unwavering dedication to the sport, there are so many different characteristics and playing statistics that make up any given team's performance. It follows then that knowing what team may win a match is entirely difficult to confidently predict beforehand.

### Why Is This Important?

A model that accurately predicts a game's outcome as well as the most important playing statistics gives teams useful insights into how to alter their strategies, training, and game performance. This will allow teams to better their chances of winning and create even more dynamic gameplay for fans across the world. For both the sake of the teams and their fans, there is much value to be found in a model that is able to analyze what statistics matter the most and can then accurately predict the outcome of any given match.

### What's Our Goal?

We are analyzing the most important statistics that correlate the strongest with winning a soccer match and using the data from previous World Cup's (2010, 2014, and 2018) to predict the outcomes of future soccter matches.

# **Data**

Our data was originally obtained from Zenodo and consisted of three individual datasets corresponding to the past 3 World Cup games in 2010, 2014, and 2018 respectively. Each dataset originally contained hundreds of datapoints, each one representing a player that participated in that World Cup, with the features being their playing statistics. To see the original 2010 World Cup dataset, see [raw_2010.csv.](https://github.com/acpollmann/soccer-training/blob/master/raw_2010.csv)


![raw_2010.png](https://raw.githubusercontent.com/acpollmann/soccer-training/master/images/raw_2010.png)

During pre-processing, we cleaned the datasets by aggregating all the player values by teams, so that we no longer had to deal with individual players as our datapoints. We also removed some unnecessary features and included some interesting ones directly from FIFA's own site. To conclude our pre-processing, we simply combined all three respective datasets into one and added a "year" feature to every datapoint.

This entire process gave us the dataset we used to conduct the rest of our project. To see our processed and cleaned dataset, see [clean_combined_data.csv](https://github.com/acpollmann/soccer-training/blob/master/clean_combined_data.csv) as well as a preview in the image below.

![cleaned_combined.png](https://raw.githubusercontent.com/acpollmann/soccer-training/master/images/cleaned_combined.png)


### Characteristics and Features

Our final, cleaned dataset consists of 384 datapoints and 27 features. Our features include World Cup Year, Team, Match Number, and all different playing-performance characteristics for each team (IE, passes, possession, fouls committed, etc). Our dataset is arguable temporal, since it originally consisted of three respective datasets that were each corresponding to playing stats during a specific year. However, our final dataset combines these three different years and simply includes the year as one of the features.

### Our Approach

We will use a random forest classifier, which randomly selects subsets of features from the dataset and uses gini impurity metric to estimate the likelihood of an incorrect classification of the datapoint under a model trained just on that subset of features. From using this process on several random subsets of features, the model provides the feature’s impurity scores, which we will then use to select the features that greatly correlate to a proper classification of match result.

After identifying the most significant features we will use supervised learning to train a random forest model, as well as serveral other models, to predict the outcome of a match given a certain set of values for each feature.

### Why Do We Believe Our Approach Will Solve Our Problem?

We chose a random forest classifier because it performs classification jobs well even on small datasets, scales well with more data, and determines relevant features effectively. Additionally, our solutions are unique in that they combine historial data with our most-important-features analysis to try and predict results.


<iframe id="igraph" scrolling="no" style="border:none;" seamless="seamless" src="https://plotly.com/~acpoll/3.embed" height="525" width="100%"></iframe>
