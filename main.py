



import streamlit as st 
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import requests
import base64 
import pickle 
from sklearn.neighbors import NearestNeighbors
import random 
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

st.title("Spotify Recommender")

st.write("This is a Spotify playlist recommender. It uses the Spotify API to get the top 50 songs from a given playlist and then uses the Spotify API to get the top 50 songs from the artist of each song. It then combines all of the songs into a new playlist. You can use the sidebar to select a playlist and then click the button to generate a new playlist.")

src = 'https://open.spotify.com/embed/playlist'

token_auth='BQA7ruNgdN-k8i13WXI2NV8F7MehddfswP67TEvd1kx0L7xzLJWRGQIXRFWWQLKbQ-AGuNkRZGeJBh2GDEG3ooSrkAGTBNoBCca3XRmWqSHP4ymyRNjP7RXPZ-sPL8_YuxsPr3-CGr3q4GHSlHEjetBXrU4pXy93t5zcrP8jbI8RCzXltVfGGc0u9B8otOkMpiFKOnbMprgPN3dukA'

client_id = 'e92a0d662da14fef8b547c2398c94b3e'
client_secret = '5beccf7349df40adb02bab1d51fe2a1c'

# Load the cosine similarity matrix
similarity = pickle.load(open('sim.pkl','rb'))
new_df = pickle.load(open('frame.pkl','rb'))


display_tuple = [ ]
all_done = False 
playlist_my = []
def refresh_token(client_id, client_secret, refresh_token):
    headers = {
        "Authorization": "Basic " + base64.b64encode(f"{client_id}:{client_secret}".encode()).decode(),
        "Content-Type": "application/x-www-form-urlencoded"
    }
    data = {
        "grant_type": "refresh_token",
        "refresh_token": refresh_token
    }
    response = requests.post("https://accounts.spotify.com/api/token", headers=headers, data=data)
    if response.status_code == 200:
        token_response = response.json()
        access_token = token_response["access_token"]
        return access_token
    else:
        raise Exception(f"Failed to refresh access token: {response.text}")

def get_playlist_id(url):
    return url.split('/')[-1].split('?')[0]

def embed_iframe(text_input):
    playlist_id = get_playlist_id(text_input)
    components.html(
        f"""
        <iframe style="border-radius:12px" src="{src}/{playlist_id}?utm_source=generator" width="100%" height="380" frameBorder="0" allowfullscreen="" allow="autoplay; clipboard-write; encrypted-media; fullscreen; picture-in-picture"></iframe>
        """, height=400
    )

def recommend_song(song_id):
    try:
        neigh = NearestNeighbors(n_neighbors=3)
        neigh.fit(similarity)
        song_index = new_df[new_df['track_id'] == song_id].index[0]
        distances, indices = neigh.kneighbors(similarity[song_index].reshape(1, -1), n_neighbors=3)
        track_ids = []
        for i in indices[0][1:]:
            track_ids.append(new_df.iloc[i]['track_id'])
        return track_ids
    except:
        audio_features_url = "https://api.spotify.com/v1/tracks/{}".format(song_id)
        headers = {
            'Authorization': f'Bearer {token_auth}'
            }
        audio_features_response = requests.get(audio_features_url, headers=headers)
        audio_features = audio_features_response.json()

        artist_id = audio_features['artists'][0]['id']

        recommended_songs_url = "https://api.spotify.com/v1/artists/{}/top-tracks?market=US".format(artist_id)
        recommended_songs_response = requests.get(recommended_songs_url, headers=headers)
        recommended_songs = recommended_songs_response.json()

        recommended_songs_ids = []
        for i in range(3):
            recommended_songs_ids.append(recommended_songs['tracks'][i]['id'])
        return recommended_songs_ids   




def get_song_details(song_id):
    try:
        fetch_by_id = f'https://api.spotify.com/v1/tracks/{song_id}'
        headers = {
            'Authorization': f'Bearer {token_auth}'
            }
        response = requests.get(fetch_by_id, headers=headers)
        response_json = response.json()

        recommended_song_detail = {}

        recommended_song_detail['name'] = response_json['name']
        recommended_song_detail['artist'] = response_json['artists'][0]['name']
        recommended_song_detail['image'] = response_json['album']['images'][0]['url']


        return recommended_song_detail

    except:
        return None

def get_audio_features(track_id):
    headers = {
        'Authorization': f'Bearer {token_auth}'
        }

    response = requests.get(f'https://api.spotify.com/v1/audio-features/{track_id}', headers=headers)
    data = response.json()

    features= [
        data['danceability'],
        data['energy'],
        data['loudness'],
        data['speechiness'],
        data['acousticness'],
        data['instrumentalness'],
        data['liveness'],
        data['valence'],
        data['tempo'],
    ]
    return features

def get_audio_features_all_song(songs):
    features = [get_audio_features(track_id) for track_id in songs]
    return features 


def fetch_songs_from_play_list(text_input):
    playlist_id = get_playlist_id(text_input)
    fetch_url = f'https://api.spotify.com/v1/playlists/{playlist_id}'
    headers = {
        'Authorization': f'Bearer {token_auth}'
        }
    response = requests.get(fetch_url, headers=headers)

    # If Error of 401 Unauthorized then refresh token
    if response.status_code == 401:
        refresh_token(client_id, client_secret, token_auth)
        response = requests.get(fetch_url, headers=headers)

    response_json = response.json()
    items = response_json['tracks']['items']
    songs = []
    for item in items:
        songs.append(item['track']['id'])
    
    return songs 


# def audio_fetures_df(songs):
#     features = get_audio_features_all_song(songs)
#     df = pd.DataFrame(features, index=songs, columns=[
#         'danceability',
#         'energy',
#         'loudness',
#         'speechiness',
#         'acousticness',
#         'instrumentalness',
#         'liveness',
#         'valence',
#         'tempo',
#     ])
#     return df


# def perform_knn(df):
#     print('Performing KNN')
#     scaler = MinMaxScaler()
#     scaled_features = scaler.fit_transform(df)
#     similarity = cosine_similarity(scaled_features)

#     model_knn = NearestNeighbors(n_neighbors=6)
#     model_knn.fit(scaled_features)

#     distances, indices = model_knn(scaled_features)

#     return distances, indices, similarity


# def recommend_knn(track_id, indices, df):

#     index = df.index.get_loc(track_id)
#     recommendations = indices[index][1:]

#     return [df.index[i] for i in recommendations] 

# def recomm(song_id):
#     try:
#         for track in playlist_my:
#             print(f"Recommendations for {track}: {recommend_knn(track)}")

        
#     except:
#         return None

def create_grid():
    # Count number of rows to create according to the the display_tuple
    rows = len(display_tuple) // 3
    if len(display_tuple) % 3 != 0:
        rows += 1
    # Create a grid
    grid = st.columns(3)
    # Create a counter
    counter = 0
    for row in range(rows):
        for col in range(3):
            if counter < len(display_tuple):
                grid[col].image(display_tuple[counter][2], width=200)
                grid[col].write(display_tuple[counter][0])
                grid[col].write(display_tuple[counter][1])
                counter += 1
            else:
                break

    



text_input = st.text_input(
    "Enter your PlayList Url 👇",

)
if text_input:
    if text_input.startswith('https://open.spotify.com/playlist/'):
        # Disable the button
        embed_iframe(text_input)
        submit_button = st.button("Recommend Songs")
        if submit_button:
            with st.spinner("Waiting for response..."):
                song_ids = fetch_songs_from_play_list(text_input)
                if song_ids:
                    ramdpm_sample = random.sample(song_ids,5)
                    for song in range(5):
                        recommended_songs = recommend_song(ramdpm_sample[song])
                        for song_recom in recommended_songs:
                            recommended_song_detail = get_song_details(song_recom)
                            if recommended_song_detail:
                                # Create Tuple and append to display_tuple
                                display_tuple.append((recommended_song_detail['name'], recommended_song_detail['artist'], recommended_song_detail['image']))
                            
                            all_done = True
                            st.spinner("Done!")
        if all_done:
            create_grid()
            
    else:
        st.write('Please enter a valid playlist url')
        submit_button = st.button("Submit", disabled=True)
# If text_input is null disable the button






st.sidebar.title("Tools")

st.sidebar.button("Generate New Playlist")
st.sidebar.button("Generate Names for Playlist")
st.sidebar.button("Detect Personality Traits")