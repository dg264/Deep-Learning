import numpy as np
import pandas as pd
import torch
import pickle
import nltk
from nltk.stem.lancaster import LancasterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import json
import tensorflow
import torch.nn as nn
import torch.nn.functional as F

class Dataframe :
    
    def __init__(self):
        # Reading the ratings data
        self.ratings = pd.read_csv('D:\Dataset/ratings.csv')
    
        #Just taking the required columns
        self.ratings = self.ratings[['userId', 'movieId','rating']]
    
        # Checking if the user has rated the same movie twice, in that case we just take max of them
        self.ratings_df = self.ratings.groupby(['userId','movieId']).aggregate(np.max)
        
        #reading the movies dataset
        self.movie_list = pd.read_csv('D:\Dataset/movies.csv')
        
        # reading the tags dataset
        self.tags = pd.read_csv('D:\Dataset/tags.csv')
        
        # inspecting various genres
        self.genres = self.movie_list['genres']
        
        self.genre_list = ""
        for index,row in self.movie_list.iterrows():
             self.genre_list += row.genres + "|"
                
        #split the string into a list of values
        self.genre_list_split = self.genre_list.split('|')
        
        #de-duplicate values
        self.new_list = list(set(self.genre_list_split))
        
        #remove the value that is blank
        self.new_list.remove('')
        
        #Enriching the movies dataset by adding the various genres columns.
        self.movies_with_genres = self.movie_list.copy()

        for genre in self.new_list :
            self.movies_with_genres[genre] = self.movies_with_genres.apply(lambda _:int(genre in _.genres), axis = 1)
            
        #Calculating the sparsity
        self.no_of_users = len(self.ratings['userId'].unique())
        self.no_of_movies = len(self.ratings['movieId'].unique())

        self.sparsity = round(1.0 - len(self.ratings)/(1.0*(self.no_of_movies * self.no_of_users)),3)
        
        # Finding the average rating for movie and the number of ratings for each movie
        self.avg_movie_rating = pd.DataFrame(self.ratings.groupby('movieId')['rating'].agg(['mean','count']))
        #self.avg_movie_rating['movieId']= self.avg_movie_rating.index
        
        #Get the average movie rating across all movies 
        self.avg_rating_all=self.ratings['rating'].mean()
        self.avg_rating_all
        #set a minimum threshold for number of reviews that the movie has to have
        self.min_reviews=30
        self.movie_score = self.avg_movie_rating.loc[self.avg_movie_rating['count']>self.min_reviews]
        
        #merging ratings and movies dataframes
        self.ratings_movies = pd.merge(self.ratings , self.movie_list, on = 'movieId')
        
        self.flag = False
        
    #create a function for weighted rating score based off count of reviews
    def weighted_rating(self , x , m = None, C = None):
        
        if m is None:
            m = self.min_reviews
            
        if C is None:
            C = self.avg_rating_all
        v = x['count']
        R = x['mean']
        
        # Calculation based on the IMDB formula
        return (v/(v+m) * R) + (m/(m+v) * C)
    
    # Gives the best movies according to genre based on weighted score which is calculated using IMDB formula
    def best_movies_by_genre(self , genre , top_n):
        
        #Calculating the weighted score for each movie
        if self.flag is not True:
            self.movie_score['weighted_score'] = self.movie_score.apply(self.weighted_rating, axis=1)
            
            #join movie details to movie ratings
            
            self.movie_score = pd.merge(self.movie_score,self.movies_with_genres,on='movieId')
            self.flag = True
        
        return pd.DataFrame(self.movie_score.loc[(self.movie_score[genre]==1)].sort_values(['weighted_score'],ascending = 
                                                                False)[['title','count','mean','weighted_score']][:top_n])
    
    #Gets the other top 10 movies which are watched by the people who saw this particular movie
    def get_other_movies(self , movie_name , top_n):
        #get all users who watched a specific movie
        df_movie_users_series = self.ratings_movies.loc[self.ratings_movies['title']==movie_name]['userId']
        
        #convert to a data frame
        df_movie_users = pd.DataFrame(df_movie_users_series,columns=['userId'])
        
        #get a list of all other movies watched by these users
        other_movies = pd.merge(df_movie_users,self.ratings_movies,on='userId')
        
        #get a list of the most commonly watched movies by these other user
        other_users_watched = pd.DataFrame(other_movies.groupby('title')['userId'].count()).sort_values('userId',ascending = 
                                                                                                        False)
        
        other_users_watched['perc_who_watched'] = round(other_users_watched['userId']*100/other_users_watched['userId'][0],1)
        
        return other_users_watched[:top_n]
    
df = Dataframe()

class Dataset:
    
    def __init__(self):
        
        with open("D:\Movie Recommender Chat Bot/recommendintents.json") as file:
            self.filedata = json.load(file)

        self.words = []
        
        self.labels = []
        
        self.docs_x = []
        
        self.docs_y = []
        
        self.stemmer = LancasterStemmer()
        
        self.stop_words = set(stopwords.words('english')) 

        for intent in self.filedata['intents']:
            
            for pattern in intent['patterns']:
                
                wrds = word_tokenize(pattern)
                wrds = [w for w in wrds if w not in self.stop_words]
                self.words.extend(wrds)
                self.docs_x.append(wrds)
                self.docs_y.append(intent["tag"])
        
            if intent['tag'] not in self.labels:
                self.labels.append(intent['tag'])

        self.words = [self.stemmer.stem(w.lower()) for w in self.words if w != "?"]
        self.words = sorted(list(set(self.words)))


        self.labels = sorted(self.labels)
        
    def get_training_data(self):
        
        training = []
        
        output = []

        out_empty = [0 for _ in range(len(self.labels))]

        for x, doc in enumerate(self.docs_x):
            bag = []

            wrds = [self.stemmer.stem(w.lower()) for w in doc]

            for w in self.words:
                if w in wrds:
                    bag.append(1)
                else:
                    bag.append(0)

            output_row = out_empty[:]
            output_row[self.labels.index(self.docs_y[x])] = 1

            #dataset.append((bag , output_row))
            training.append(bag)
            output.append(output_row)
            
        training = np.array(training)
        
        output = np.array(output)
        
        from sklearn.utils import shuffle
        
        training , output = shuffle(training , output , random_state = 0)
        
        return training , output
    
    def bag_of_words(self , s):
        bag = [0 for _ in range(len(self.words))]
    
        s_words = word_tokenize(s)
        s_words = [self.stemmer.stem(word.lower()) for word in s_words]
        s_words = [w for w in s_words if w not in self.stop_words]

        for se in s_words:
            for i , w in enumerate(self.words):
                if w == se:
                    bag[i] = 1

        return np.array(bag)


data = Dataset()
training , output = data.get_training_data()

class Net(nn.Module):

    def __init__(self , training , output):

        super().__init__()

        self.fc1 = nn.Linear(len(training[0])  , 8)
        self.fc2 = nn.Linear(8 , 8)

        self.fc3 = nn.Linear(8 , len(output[0]))

    def forward(self , x):
        x = F.relu(self.fc1(x))

        x = F.relu(self.fc2(x))

        x = self.fc3(x)

        return F.softmax(x , dim = 1)

net = Net(training , output)

import torch.optim as optim

class Train_Model:
    
    def __init__(self  , training , output):
        
        self.optimizer = optim.Adam(net.parameters() , lr = 0.001)
        self.loss_function = nn.MSELoss()

        self.X = torch.tensor(training , dtype = torch.float32)
        self.Y = torch.tensor(output , dtype = torch.float32)
        
    def start_training(self , net , BATCH_SIZE , EPOCHS):

        for epoch in range(EPOCHS):
            
            for i in range(0 , len(self.X) , BATCH_SIZE):
                #print(i , i+BATCH_SIZE)
                batch_X = self.X[i : i + BATCH_SIZE].view(-1 , len(self.X[0]))
                batch_Y = self.Y[i : i + BATCH_SIZE]
        
                net.zero_grad()
                outputs = net(batch_X)
                loss = self.loss_function(outputs , batch_Y)
                loss.backward()
                self.optimizer.step()
            
            print(loss)
            
        return net


tr = Train_Model(training , output)

net = tr.start_training(net , 8 , 500)

import random
def chat():
    print("Start talking with the bot (type quit to stop)")
    while True:
        inp = input("You: ")

        if inp.lower() == "quit":
            break
        with torch.no_grad() :
            results = net(torch.tensor(data.bag_of_words(inp) , dtype = torch.float32).view(1 , len(data.words)))[0]
            results_index = np.argmax(np.array(results))
            tag = data.labels[results_index]
            
            if results[results_index] > 0.6:
                
                if(tag == "toprated"):
                    print("Please give me genre of the movies you want to watch :")
                    genr = input()

                    if genr not in df.new_list:
                        print("I don't have this genre")
                        continue
                        
                    print("Please tell me number of movies that i should suggest you")
                    top_n = int(input())
                    print(df.best_movies_by_genre(genr , top_n))
                    
                if(tag == "contentbasedrated"):
                    print("Please tell me name of your favorite movie :")
                    mov = input()
                    print("Please tell me number of movies that i should suggest you")
                    top_n = int(input())
                    print(df.get_other_movies(mov , top_n))
                    
                for tg in data.filedata["intents"]:
                    if tg['tag'] == tag:
                        responses = tg['responses']

                print(random.choice(responses))

            else:
                print("I did'nt get that , please ask another question.")


chat()
	