# 🔍 Fraud Analysis
This is a side project aimed at catching fraud using clustering and text mining. 

# Introduction
### Datasource: 
- Source 1- banksim: https://www.kaggle.com/datasets/ealaxi/banksim1
  (class 0: Not a fraud; class 1: a fraud)

  <img width="350" alt="image" src="https://github.com/wenchitseng/fraud_analysis/assets/145182368/6cf11f65-fc08-4cb0-bd15-9d3ee413898c">

- Source 2- enron email: https://www.kaggle.com/datasets/wcukierski/enron-email-dataset

  <img width="900" alt="image" src="https://github.com/wenchitseng/fraud_analysis/assets/145182368/f5f7746c-f960-4851-9381-66f09ffb08d9">




# ⓵ Fraud Analysis: Simple Statistic
## Group by category  
Based on the probability in the "fraud" column, the majority of fraud is observed in travel, leisure, and sports-related transactions.

<img width="307" alt="image" src="https://github.com/wenchitseng/fraud_analysis/assets/145182368/c9343a9e-2063-40fa-9c8c-783c9bce9040">

## Group by age  
The result of the Age Group 0 is a bit different from the rest. However, Age Group 0 only has 40 cases in the entire dataset, making it not feasible to split these out into a separate group and run the model on that amount of observations.

<img width="188" alt="image" src="https://github.com/wenchitseng/fraud_analysis/assets/145182368/ac324f92-4687-498b-a219-ddf2f3d920c9"> <img width="86" alt="image" src="https://github.com/wenchitseng/fraud_analysis/assets/145182368/988d6382-a0ad-4588-ade9-3d538141a06b">

# ⓶ Fraud Analysis: Clustering 
## KMeans Clustering
  1. Run **MiniBatch** Kmeans over the number of clusters
  2. Use **Elbow method** to find the optimal number of clusters (2 clusters based on the plot)
     <img width="500" alt="image" src="https://github.com/wenchitseng/fraud_analysis/assets/145182368/87c20602-df45-4fcb-a026-5f0c354a487d">

  3. Prediction: Fit the KMeans model on existing data and define a cutoff point for fraud when new data is introduced.
    (Here, exceeding the 85th percentile of the distances to the cluster centroids is considered potential fraud.)

     <img width="700" alt="image" src="https://github.com/wenchitseng/fraud_analysis/assets/145182368/a0268cc3-d07e-43c2-a498-430f4f3c5a61">


  5. Perform an ROC plot to check the model accuracy

     <img width="500" alt="image" src="https://github.com/wenchitseng/fraud_analysis/assets/145182368/3b0a2112-ba89-4931-bf32-f08f8c21ff92">

# ⓷ Fraud Analysis: Text Data
## Word Search
Search for specific words that are suspicious or appeared in previous fraud cases. This technique can be utilized in textual data such as emails, transaction descriptions, employee notes, insurance claims, etc.

## Text Mining
  1. Import NLTK packages for text preprocessing
  2. Define **stop words, punctuation, and lemmatizer**
  3. Clean textual data based on the previous step
     
     <img width="900" alt="image" src="https://github.com/wenchitseng/fraud_analysis/assets/145182368/c4956c58-1b54-4036-b53f-137aee2ff429">

  4. Run **Latent Dirichlet Allocation (LDA)** topic model
     - use **gensim**
     - This step assigns each word a unique ID and calculate its frequency in each document
  6. Decide the number of topics and extract the top 5 words for each topic

     <img width="1197" alt="image" src="https://github.com/wenchitseng/fraud_analysis/assets/145182368/1bcf6607-5157-4ea2-acc4-cf6d9c9ce331">


  7. Observe if there is an unusual topic that needs further investigation

     










