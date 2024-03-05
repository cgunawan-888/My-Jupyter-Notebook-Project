# Installing Libraries:
%pip install pymongo
%pip install matplotlib
%pip install numpy
%pip install statsmodels

# Importing Libraries:
import pymongo 
import json
import certifi
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
import statsmodels.formula.api

# ================================================================================================================

# Loading my credentials file without security breach
with open('ChrisG_Credentials.txt') as f:
    d = json.load(f)

# Connecting to the database using a well-known certificate
myclient = pymongo.MongoClient(d['my_credentials'], tlsCAFile=certifi.where()) 

# Defining my database
mydb = myclient["DATA320_Fall2023"]

# Defining the IMDB Collection's Data-Frame for the year 2014
imdb_pipeline = pd.DataFrame(mydb.Superhero.find({ "release_date": re.compile("2014") }))

# Defining the Metacritic Collection's Data-Frame for the year 2014
metacritic_view = pd.DataFrame(mydb.Metacritic.find({ "release_date": re.compile("2014") }))

# Converting release date column(s) to date & time format
imdb_pipeline.release_date = pd.to_datetime(imdb_pipeline.release_date, errors='coerce')
metacritic_view.release_date = pd.to_datetime(metacritic_view.release_date, errors='coerce')

# Sort the rows based on IMDB's date & time (from the oldest to the newest - in chronological order)
imdb_pipeline.sort_values(by='release_date', ascending=True, inplace = True)

# Converting the necessary columns to numeric format
metacritic_view.score = pd.to_numeric(metacritic_view.score, errors='coerce')
imdb_pipeline.budget = pd.to_numeric(imdb_pipeline.budget, errors='coerce')
imdb_pipeline.runtime = pd.to_numeric(imdb_pipeline.runtime, errors='coerce')
imdb_pipeline.user_rating = pd.to_numeric(imdb_pipeline.user_rating, errors='coerce')
imdb_pipeline.votes = pd.to_numeric(imdb_pipeline.votes, errors='coerce')
imdb_pipeline.opening_weekend = pd.to_numeric(imdb_pipeline.opening_weekend, errors='coerce')
imdb_pipeline.gross_sales = pd.to_numeric(imdb_pipeline.gross_sales, errors='coerce')

# Converting the necessary columns (with multiple data in a field) to arrays
imdb_pipeline.genres = imdb_pipeline["genres"].apply(lambda x: x.split(','))
imdb_pipeline.cast = imdb_pipeline["cast"].apply(lambda x: x.split(','))
imdb_pipeline.director = imdb_pipeline["director"].apply(lambda x: x.split(','))
imdb_pipeline.producer = imdb_pipeline["producer"].apply(lambda x: x.split(','))
imdb_pipeline.company = imdb_pipeline["company"].apply(lambda x: x.split(','))

# Eliminating everything after "::" in mpaa_rating column
imdb_pipeline.mpaa_rating = imdb_pipeline["mpaa_rating"].str.split("::").str[0]

# Merging IMDB and Metacritic
combine_view = pd.merge(imdb_pipeline, metacritic_view, how="inner", on="title")

# Display the data type of each column in the merged table
print(combine_view.dtypes)
print()

# Accounting Format without $ sign (budget & gross sales columns consist of multi-currencies)
pd.options.display.float_format = '{:,.0f}'.format

# Set Column Width to 30
pd.options.display.max_colwidth = 30

# Display the merged table
display(combine_view)
print()

# ================================================================================================================

# Regression Model #1
ols_model = statsmodels.formula.api.ols(
    formula="gross_sales ~ runtime + budget + votes",
    data=combine_view).fit()
print(ols_model.summary())

# ================================================================================================================

# Regression Model #2
ols_model_2 = statsmodels.formula.api.ols(
    formula="score ~ user_rating",
    data=combine_view).fit()
print(ols_model_2.summary())

# ================================================================================================================

Then, I plotted many different type of charts using matplotlib library.
