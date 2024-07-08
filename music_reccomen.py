{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Attempt 2 (Succesful)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "c:\\Users\\DELL\\anaconda3\\Lib\\site-packages\\sklearn\\utils\\validation.py:767: FutureWarning: is_sparse is deprecated and will be removed in a future version. Check `isinstance(dtype, pd.SparseDtype)` instead.\n",
      "  if not hasattr(array, \"sparse\") and array.dtypes.apply(is_sparse).any():\n",
      "c:\\Users\\DELL\\anaconda3\\Lib\\site-packages\\sklearn\\utils\\validation.py:605: FutureWarning: is_sparse is deprecated and will be removed in a future version. Check `isinstance(dtype, pd.SparseDtype)` instead.\n",
      "  if is_sparse(pd_dtype):\n",
      "c:\\Users\\DELL\\anaconda3\\Lib\\site-packages\\sklearn\\utils\\validation.py:614: FutureWarning: is_sparse is deprecated and will be removed in a future version. Check `isinstance(dtype, pd.SparseDtype)` instead.\n",
      "  if is_sparse(pd_dtype) or not is_extension_array_dtype(pd_dtype):\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      ".............RECOMMENDED SONGS...............\n",
      "\n",
      "Song: happier\n",
      "Artist(s): Olivia Rodrigo\n",
      "\n",
      "Song: snow on the beach (feat. lana del rey)\n",
      "Artist(s): Taylor Swift, Lana Del Rey\n",
      "\n",
      "Song: bored\n",
      "Artist(s): Billie Eilish\n",
      "\n",
      "Song: until i found you\n",
      "Artist(s): Stephen Sanchez\n",
      "\n",
      "Song: golden hour\n",
      "Artist(s): JVKE\n",
      "\n",
      "Song: when i r.i.p.\n",
      "Artist(s): Labrinth\n",
      "\n",
      "Song: i hate u\n",
      "Artist(s): SZA\n",
      "\n",
      "Song: still don't know my name\n",
      "Artist(s): Labrinth\n",
      "\n",
      "Song: tv\n",
      "Artist(s): Billie Eilish\n",
      "\n"
     ]
    }
   ],
   "source": [
    "import pandas as pd\n",
    "from sklearn.feature_extraction.text import TfidfVectorizer\n",
    "from sklearn.neighbors import NearestNeighbors\n",
    "\n",
    "# Read the CSV file into a DataFrame\n",
    "df = pd.read_csv(\"E:\\\\Practies dataset\\\\spotify-2023.csv\", encoding=\"latin1\")\n",
    "\n",
    "\n",
    "\n",
    "# Drop duplicate rows and rows with missing values\n",
    "# df.drop_duplicates(inplace=True)\n",
    "df.dropna(inplace=True)\n",
    "df.reset_index(drop=True, inplace=True)\n",
    "df.track_name = df.track_name.map(str.lower)\n",
    "\n",
    "\n",
    "\n",
    "\n",
    "\n",
    "# Concatenate track name and artist name for text features\n",
    "text_features = df['track_name'] + ' ' + df['artist(s)_name']\n",
    "\n",
    "# i am Useing  TF-IDF to convert text features into numerical vectors\n",
    "tv = TfidfVectorizer()\n",
    "tfidf_matrix = tv.fit_transform(text_features)\n",
    "\n",
    "# Select additional features for the model\n",
    "additional_features = df[[\"danceability_%\", \"valence_%\", \"energy_%\", \"acousticness_%\", \"instrumentalness_%\",\n",
    "                          \"liveness_%\", \"speechiness_%\", \"in_spotify_playlists\", \"in_spotify_charts\",\n",
    "                          \"in_apple_playlists\", \"in_apple_charts\"]].values\n",
    "\n",
    "# Concatenate TF-IDF matrix with additional features\n",
    "matrix = pd.concat([pd.DataFrame(tfidf_matrix.toarray()), pd.DataFrame(additional_features)], axis=1)\n",
    "\n",
    "# Train Nearest Neighbors model\n",
    "model = NearestNeighbors(metric='cosine')\n",
    "model.fit(matrix)\n",
    "\n",
    "try:\n",
    "    mv = input(\"Enter Music name: \").lower()\n",
    "    mi = df[df[\"track_name\"] == mv].index[0]\n",
    "    dis, idx = model.kneighbors(matrix.iloc[mi].values.reshape(1, -1), n_neighbors=10)\n",
    "\n",
    "    # Print similar tracks\n",
    "    similar_tracks = df.loc[idx[0][1:],[\"track_name\",\"artist(s)_name\"]]\n",
    "    # for i in similar_tracks:\n",
    "    #     print(i)\n",
    "\n",
    "    print(\".............RECOMMENDED SONGS...............\")\n",
    "    print()\n",
    "    for index, row in similar_tracks.iterrows():\n",
    "        print(\"Song:\", row[\"track_name\"])\n",
    "        print(\"Artist(s):\", row[\"artist(s)_name\"])\n",
    "        print() \n",
    "\n",
    "except IndexError:\n",
    "    print(\"Music not found in the dataset.\")\n",
    "\n",
    "except Exception as e:\n",
    "    print(\"An error occurred:\", e)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "base",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
