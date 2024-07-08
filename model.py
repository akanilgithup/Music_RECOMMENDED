import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# Read the CSV file into a DataFrame
df = pd.read_csv("E:\\Practies dataset\\spotify-2023.csv", encoding="latin1")



# Drop duplicate rows and rows with missing values
# df.drop_duplicates(inplace=True)
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)
df.track_name = df.track_name.map(str.lower)





# Concatenate track name and artist name for text features
text_features = df['track_name'] + ' ' + df['artist(s)_name']

# i am Useing  TF-IDF to convert text features into numerical vectors
tv = TfidfVectorizer()
tfidf_matrix = tv.fit_transform(text_features)

# Select additional features for the model
additional_features = df[["danceability_%", "valence_%", "energy_%", "acousticness_%", "instrumentalness_%",
                          "liveness_%", "speechiness_%", "in_spotify_playlists", "in_spotify_charts",
                          "in_apple_playlists", "in_apple_charts"]].values

# Concatenate TF-IDF matrix with additional features
matrix = pd.concat([pd.DataFrame(tfidf_matrix.toarray()), pd.DataFrame(additional_features)], axis=1)

# Train Nearest Neighbors model
model = NearestNeighbors(metric='cosine')
model.fit(matrix)

try:
    mv = input("Enter Music name: ").lower()
    mi = df[df["track_name"] == mv].index[0]
    dis, idx = model.kneighbors(matrix.iloc[mi].values.reshape(1, -1), n_neighbors=10)

    # Print similar tracks
    similar_tracks = df.loc[idx[0][1:],["track_name","artist(s)_name"]]
    # for i in similar_tracks:
    #     print(i)

    print(".............RECOMMENDED SONGS...............")
    print()
    for index, row in similar_tracks.iterrows():
        print("Song:", row["track_name"])
        print("Artist(s):", row["artist(s)_name"])
        print() 

except IndexError:
    print("Music not found in the dataset.")

except Exception as e:
    print("An error occurred:", e)
