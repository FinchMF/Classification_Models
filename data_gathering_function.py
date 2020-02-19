#############
# FUNCTIONS #
#############

import requests
import json



# call spotify recieve playlist of songs as dictionary
def call_spotify(str):
    i = str
    base = 'https://api.spotify.com/v1/playlists/'
    end = '/tracks'
    r = requests.get(base + i + end, headers=headers)
    songs = r.json()
    return songs

