#!/usr/bin/env python
# coding: utf-8

# ## Import Data

# In[2]:


import pandas as pd


# In[3]:


# Load the dataset
df = pd.read_csv('/Users/cengwenqi/Library/CloudStorage/OneDrive-UCIrvine/banksim.csv')
df.head()


# # Fraud Analysis: Simple Statistic

# In[4]:


df.drop(['Unnamed: 0'], axis=1, inplace=True)


# In[5]:


print(df.shape)
df.head()


# #### Group by transcation category

# In[8]:


# Ｇroup by category and take the mean of the 'amount' and 'fraud' columns
df.groupby('category')[['amount','fraud']].mean()


# **Insights: Based on the results, the majority of fraud is observed in travel, leisure and sports related transactions.**

# #### Group by age

# In[120]:


# Ｇroup by age and take the mean of the 'amount' and 'fraud' columns
df.groupby('age')[['amount','fraud']].mean()


# In[121]:


# Count the values of the observations in each age group
print(df['age'].value_counts())


# **Insight: The result of the Age Group 0 is a bit different from the rest. However, the Age Group 0 only has 40 cases in the entire dataset, making it not feasible to split these out into a separate group and run the model on that amount of observations.**

# #### Define Normal Behavior

# In[12]:


import matplotlib.pyplot as plt


# In[124]:


# Create two dataframes with fraud and non-fraud data 
df_fraud = df.loc[df.fraud == 1] 
df_non_fraud = df.loc[df.fraud == 0]


# In[125]:


# Plot histograms of the amounts in fraud and non-fraud data 
plt.hist(df_fraud.amount, alpha=0.5, label='fraud')
plt.hist(df_non_fraud.amount, alpha=0.5, label='nonfraud')
plt.legend()
plt.show()


# # Fraud Analysis: Clustering

# ## KMeans 

# In[9]:


from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
import numpy as np


# #### Convert value to float and scale the data

# **We scale the data here to prevent the results from being dominated by the majority.**

# In[10]:


# Convert 'amount' column to float
df['amount'] = pd.to_numeric(df['amount'], errors='coerce')

# Initialize the MinMaxScaler
scaler = MinMaxScaler()

# Scale the 'amount' column
# Reshape is needed because it expects 2D array
X_scaled = scaler.fit_transform(df[['amount']])

print(X_scaled)


# #### Use Elbow method to find the optimal number of cluster

# In[13]:


# Define the range of clusters to try
clustno = range(1, 5)

# Run MiniBatch Kmeans over the number of clusters
kmeans = [MiniBatchKMeans(n_clusters=i, random_state=0) for i in clustno]

# Obtain the score for each model
score = [kmeans[i].fit(X_scaled).score(X_scaled) for i in range(len(kmeans))]

# Plot the models and their respective score 
plt.plot(clustno, score)
plt.xlabel('Number of Clusters')
plt.ylabel('Score')
plt.title('Elbow Curve')
plt.show()


# **Insights: The optimal number of clusters should probably be at around 2 clusters, as that is where the elbow is in the curve.**

# #### Run K-Mean model and define a cutoff point for fraud

# In[133]:


# Define the variable 'y' with appropriate values
y = df['fraud']

# Split the data into training and test set
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)


# In[134]:


# Define K-means model
kmeans = KMeans(n_clusters=2, random_state=0).fit(X_train)

# Get the cluster number for each datapoint
X_test_clusters = kmeans.predict(X_test)

# Save cluster centriods
X_test_clusters_centers = kmeans.cluster_centers_


# In[135]:


# Calculate the distance to the cluster centroids for each point
dist = [np.linalg.norm(x-y) for x, y in zip(X_test, X_test_clusters_centers[X_test_clusters])]

# Create predictions based on the distance(we set the threshold to 0.85)
km_y_pred = np.array(dist)
km_y_pred[dist >= np.percentile(dist, 85)] = 1
km_y_pred[dist < np.percentile(dist, 85)] = 0


# #### Check accuracy of catching fraud using confusion metrix

# In[136]:


# Obtain the ROC score
print(roc_auc_score(y_test, km_y_pred))

# Create a confusion matrix
km_cm = confusion_matrix(y_test, km_y_pred)

# Plot the confusion matrix in a figure to visualize results 
# Confusion matrix
ConfusionMatrixDisplay.from_predictions(y_test, km_y_pred, cmap = plt.cm.Blues, normalize = None, display_labels = ['0', '1'])


# ## DBSCAN

# In[6]:


from sklearn.cluster import DBSCAN


# In[14]:


# Initialize and fit the DBSCAN model
db = DBSCAN(eps=0.9, min_samples=10, n_jobs=-1).fit(X_scaled)

# Obtain the predicted labels and calculate number of clusters
pred_labels = db.labels_
n_clusters = len(set(pred_labels)) - (1 if -1 in pred_labels else 0)

# Print performance metrics for DBSCAN
print('Estimated number of clusters: %d' % n_clusters)


# In[15]:


# Count observations in each cluster number
counts = np.bincount(pred_labels[pred_labels >= 0])

# Print the result
print(counts)


# **Insights: The presence of only one significant cluster suggests that there might not be any outliers. This indicates that fraud detection may require a more nuanced approach or the inclusion of more observations. If there is more than one cluster, the cluster with the fewest observations could indicate rare cases or potential fraud that needs further investigation.**

# # Fraud Analysis: Text Mining

# In[49]:


# Load the dataset
df = pd.read_csv('/Users/cengwenqi/Library/CloudStorage/OneDrive-UCIrvine/enron_emails.csv')
df.head()


# ## Word Search

# #### Search for the email containing specific word

# In[22]:


# Create a list of terms to search for
searchfor = ['enron stock', 'sell stock', 'stock bonus', 'sell enron stock']

filtered_emails = df['clean_content'].str.contains('|'.join(searchfor), na=False) # "|" represents "OR"

# Select rows that content 'sell enron stock'
print(df.loc[filtered_emails])


# #### Flag fraud based on word search

# In[26]:


# Create flag variable where the emails match the searchfor terms
df['flag'] = np.where((df['clean_content'].str.contains('|'.join(searchfor)) == True), 1, 0)

# Count the values of the flag variable
count = df['flag'].value_counts()
print(count)

# total_observation = df['flag'].count()
# print(total_observation)


# ## Text Mining

# In[35]:


# Import nltk packages and string 
from nltk.corpus import stopwords
import string, nltk

from nltk.stem.wordnet import WordNetLemmatizer


# In[40]:


nltk.download('stopwords')
nltk.download('wordnet')


# ### Define stopwords, punctuation, and convert words to base or root

# In[74]:


# Define stopwords to exclude
stop = set(stopwords.words('english'))
stop.update(("to","cc","subject","http","from","sent", "ect", "u", "fwd", "www", "com"))

# Define punctuations to exclude and lemmatizer
exclude = set(string.punctuation)

# Convert words
lemma = WordNetLemmatizer()


# ### Clean text data

# In[75]:


# Define word cleaning function
def clean(text, stop):
    text = text.rstrip()
	# Remove stopwords
    stop_free = " ".join([word for word in text.lower().split() if ((word not in stop) and (not any(char.isdigit() for char in word)))])
	# Remove punctuations
    punc_free = ''.join(word for word in stop_free if word not in exclude)
	# Lemmatize all words
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())      
    return normalized


# In[76]:


text_clean=[]
for text in df['content']:
    text_clean.append(clean(text, stop).split())    


text_clean


# ### Run Latent Dirichlet Allocation (LDA) topic model

# In[77]:


import gensim
from gensim import corpora


# In[78]:


# Define the dictionary
dictionary = corpora.Dictionary(text_clean)

# Define the corpus
corpus = [dictionary.doc2bow(text) for text in text_clean]

# Print corpus and dictionary
print(dictionary)
print(corpus)


# In[79]:


# Define the LDA model
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=5, id2word=dictionary, passes=5)
#Use gensim.models. to select the LDA model

# Save the topics and top 5 words
topics = ldamodel.print_topics(num_words=5)

# Print the results
for topic in topics:
    print(topic)


# In[ ]:




