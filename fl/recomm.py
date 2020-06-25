import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#df=pd.read_csv('Book.csv')


def recom(movie_user_likes):
	###### helper functions. Use them when needed #######

	def get_title_from_index(index):
		return df[df.index == index]["Title"].values[0]

	def get_index_from_title(Title):
		return df[df.Title == Title]["index"].values[0]
	##################################################

	##Step 1: Read CSV File
	books = pd.read_csv("../BEP/Bookz.csv")
	books=books[:1000]
	df=books
	img=pd.read_csv("Imagez.csv")
	##Step 2: Select Features

	features = ['Title','Author','Publisher']
	##Step 3: Create a column in DF which combines all selected features
	for feature in features:
		df[feature] = df[feature].fillna('')

	def combine_features(row):
		try:
			return row['Title'] +" "+row['Author']+" "+row['Publisher']
		except:
			print("Error:", row)

	df["combined_features"] = df.apply(combine_features,axis=1)

	#print "Combined Features:", df["combined_features"].head()

	##Step 4: Create count matrix from this new combined column
	cv = CountVectorizer()

	count_matrix = cv.fit_transform(df["combined_features"])

	##Step 5: Compute the Cosine Similarity based on the count_matrix
	cosine_sim = cosine_similarity(count_matrix) 

	## Step 6: Get index of this movie from its title
	movie_index = get_index_from_title(movie_user_likes)

	similar_movies = list(enumerate(cosine_sim[movie_index]))

	## Step 7: Get a list of similar movies in descending order of similarity score
	sorted_similar_movies = sorted(similar_movies,key=lambda x:x[1],reverse=True)

	## Step 8: Print titles of first 50 movies
	l=[]
	t=[]
	i=0
	for element in sorted_similar_movies:
			l.append(get_title_from_index(element[0]))
			t.append(get_index_from_title(l[i]))
			i=i+1
			if i>9:
				break

	output=l
	index=t

	imgg=[]
	year=[]
	author=[]
	final_list=[]
	for i in index:
		imgg.append(img["Image-URL-M"][i-1])
		year.append(books["Year"][i-1])
		author.append(books["Author"][i-1])
	for i in range(len(index)):
		temp=[]
		temp.append(output[i])
		temp.append(imgg[i])
		temp.append(year[i])
		temp.append(author[i])
		final_list.append(temp)
	return final_list




def bookdisp():
	books=pd.read_csv("Bookz.csv")
	img=pd.read_csv("Imagez.csv")

	title=[]
	imgg=[]
	year=[]
	author=[]
	finallist=[]

	r=np.random.randint(2,1000,10)

	for i in r:
		title.append(books["Title"][i-1])
		imgg.append(img["Image-URL-M"][i-1])
		year.append(books["Year"][i-1])
		author.append(books["Author"][i-1])

	for i in range(10):
		temp=[]
		temp.append(title[i])
		temp.append(imgg[i])
		temp.append(year[i])
		temp.append(author[i])
		finallist.append(temp)

	return finallist