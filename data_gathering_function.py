#############
# FUNCTIONS #
#############

import requests
import json
import config


# retrieve access tokens
def get_tokens():
    client_id = config.client_id
    client_secret = config.client_secret

    grant_type = 'client_credentials'

    body_params = {'grant_type' : grant_type}

    url='https://accounts.spotify.com/api/token'

    response=requests.post(url, data=body_params, auth = (client_id, client_secret)) 
    response.text
    token = eval(response.text)
    print('Full Token Data')
    print(token)
    print('\n')
    print('token needed:', token.get('access_token'))
    
    return token

token = get_tokens()

access_token = token.get('access_token')
headers = {'Authorization': 'Bearer '+ access_token}

# dictionary of playlist Ids

list_of_genre_ids = ['7Mr3wEdKgaiAP4Cm2a6vda', 
                     '4MRGjKqlWuJZJ8XHOGcqkR', 
                     '37i9dQZF1DX4dyzvuaRJ0n',
                     '1o1HVRGIxwCcdSnNnZ69IC',
                     '37i9dQZF1DWWEJlAGA9gs0',
                     '0TCtFMz5lY6jTfusk66ZFj',
                     '3kTtdRE1CtRyRKdicfOGAR',
                     '30BUPgw52SWNm2ZWZZc86A',
                     '6Ph1K0QWCcEwYRr0VhVt6C',
                     '5khoF3ksobwfVwOazDqpqI',
                     '6mRRGF4klfgUzbD2ZKOCq0',
                     '37i9dQZF1DWZd79rJ6a7lp',
                     '37i9dQZF1DX0xLQsW8b5Zx',
                     '2SwjQPegrTTYaOsWQrwhMe'
                    ]

list_of_genre_playlists = ['hip hop playlist',
                           'post rock playlist',
                           'electronic playlist',
                           'detroit_techo playlist',
                           'classical playlist',
                           'disco playlist',
                           'electro indie pop playlist',
                           'industrial pop playlist',
                           'french playlist',
                           'spanish playlist',
                           'ska playlist',
                           'sleep playlist',
                           'rockabilly playlist',
                           '50s Hits playlist'
                           ]

genre_dict = {list_of_genre_playlists[i]: list_of_genre_ids[i] for i in range(len(list_of_genre_playlists))}
genre_dict

import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm

#####################
# CALL FUNCTION 1.1 #
#####################

# call spotify recieve playlist of songs as dictionary
def call_spotify(str):
    i = str
    base = 'https://api.spotify.com/v1/playlists/'
    end = '/tracks'
    r = requests.get(base + i + end, headers=headers)
    songs = r.json()
    return songs


#####################
# CALL FUNCTION 1.2 #
#####################


# fetch id function
def fetch_ids(songs):
    ids = []
    for song in songs['items']:
        ids.append(song['track']['id'])
    return ids

# fetch audio feature links function
def audio_features_(data):
    base =  'https://api.spotify.com/v1/audio-features/'
    list_of_ids = []
    for i in data:
        list_of_ids.append(base + str(i))
    return list_of_ids

# retrieve dictionary of track feature data
def fetch_features(links):
    feature_data = []
    for i in links:
        x = requests.get(i, headers=headers)
        j = x.content
        feature_data.append(json.loads(j))
    return feature_data

# combing the three functions into one

# all functions together to pull song list in and return dataframe
def song_return_feature_df(songs):
    ids = fetch_ids(songs)
    links = audio_features_(ids)
    feature_list = fetch_features(links)
    df = pd.DataFrame(feature_list)
    return df


#####################
# CALL FUNCTION 1.3 #
#####################

# call spotify playlist, return features as dataframe
def call_spotify_return_feat_df(genre_id):
    songs = call_spotify(genre_id)
    df = song_return_feature_df(songs)
    return df



#####################
# CALL FUNCTION 1.4 #
#####################

# retrieving track titles from playlist
def fetch_titles(songs):
    titles = []
    for song in songs['items']:
        titles.append(song['track']['name'])
    return titles

# calls playlist, returns titles of songs in playlist as dataframe
def fetch_track_names(genre_id):
    songs = call_spotify(genre_id)
    titles = fetch_titles(songs)
    df = pd.DataFrame(titles)
    df = df.rename(columns={0: 'Song Titles'})
    return df


#####################
# CALL FUNCTION 1.5 #
#####################

# retrieving track tempos 
def get_tempo(song_id):
    base = 'https://api.spotify.com/v1/audio-analysis/'
    link = base + song_id 
    r = requests.get(link, headers=headers)
    r_j = r.json()
    track_tempo = r_j['track']['tempo']
    return track_tempo

# retrieving all track tempos
def get_all_tempos(genre_id):
    songs = call_spotify(genre_id)
    ids = fetch_ids(songs)
    tempos = []
    for i in tqdm(ids):
        tempos.append(get_tempo(i))
    return tempos

# retrieving track tempo confidence
def get_tempo_consistency(song_id):
    base = 'https://api.spotify.com/v1/audio-analysis/'
    link = base + song_id 
    r = requests.get(link, headers=headers)
    r_j = r.json()
    tempo_confidence = r_j['track']['tempo_confidence']
    return tempo_confidence


# convert key number to pitch name
def convert_key_to_pitchname(track_key):
    if track_key == -1:
        return 'Key Not Defined'
    if track_key == 0:
        return 'C'
    if track_key == 1:
        return 'C#/Db'
    if track_key == 2:
        return 'D'
    if track_key == 3:
        return 'D#/Eb'
    if track_key == 4:
        return 'E'
    if track_key == 5:
        return 'F'
    if track_key == 6:
        return 'F#/Gb'
    if track_key == 7:
        return 'G'
    if track_key == 8:
        return 'G#/Ab'
    if track_key == 9:
        return 'A'
    if track_key == 10:
        return 'A#/Bb'
    if track_key == 11:
        return 'B'

# retrieving track key
def get_key(song_id):
    base = 'https://api.spotify.com/v1/audio-analysis/'
    link = base + song_id 
    r = requests.get(link, headers=headers)
    r_j = r.json()
    track_key = r_j['track']['key']
    converted_key = convert_key_to_pitchname(track_key)
    return converted_key

# retrieving all track keys from playlist
def get_all_keys(genre_id):
    songs = call_spotify(genre_id)
    ids = fetch_ids(songs)
    keys = []
    count = 0
    for i in ids:
        keys.append(get_key(i))
        print('--- key data {} retrieved ---'.format(count))
        count += 1
    return keys

# retrieving track key confidence
def get_key_consistency(song_id):
    base = 'https://api.spotify.com/v1/audio-analysis/'
    link = base + song_id 
    r = requests.get(link, headers=headers)
    r_j = r.json()
    key_confidence = r_j['track']['key_confidence']
    return key_confidence

# convert mode to diatonic name
def convert_mode_to_diatonic_name(track_mode):
    if track_mode == -1:
        return 'Mode Not Defined'
    if track_mode == 0:
        return 'Minor'
    if track_mode == 1:
        return 'Major'


# retrieving track mode of key (maj or min)
def get_key_mode(song_id):
    base = 'https://api.spotify.com/v1/audio-analysis/'
    link = base + song_id
    r = requests.get(link, headers=headers)
    r_j = r.json()
    track_mode = r_j['track']['mode']
    converted_mode = convert_mode_to_diatonic_name(track_mode)
    return converted_mode

# retrieving all track key modes from playlist
def get_all_key_modes(genre_id):
    songs = call_spotify(genre_id)
    ids = fetch_ids(songs)
    modes = []
    count = 0
    for i in ids:
        modes.append(get_key_mode(i))
        print('--- mode data {} retrieved ---'.format(count))
        count += 1
    return modes

# retrieveing track key mode confidence
def get_mode_consistency(song_id):
    base = 'https://api.spotify.com/v1/audio-analysis/'
    link = base + song_id 
    r = requests.get(link, headers=headers)
    r_j = r.json()
    mode_confidence = r_j['track']['mode_confidence']
    return mode_confidence

#######################
# FEATURE ENGINEERING #
#######################

# get audio analysis links
def audio_analysis_links(song_ids):
    base =  'https://api.spotify.com/v1/audio-analysis/'
    list_of_audio_analysis_links = []
    for i in tqdm(song_ids):
        list_of_audio_analysis_links.append(base + str(i))
    return list_of_audio_analysis_links


# retrieve dictionary of track analysis data
def fetch_analysis(links):
    analysis_data = []
    count = 0
    for i in links:
        r = requests.get(i, headers=headers)
        r_j = r.content
        analysis_data.append(json.loads(r_j))
        print('--- data on link {} retrieved--'.format(count))
        count += 1
    return analysis_data

# keys of section (chord progression) in track
def get_harmonic_progression(song):
    section_keys = []
    for section in song['sections']:
        k = section['key']
        if k == -1:
            section_keys.append('Key not defined')
        if k == 0:
            section_keys.append('C')
        if k == 1:
            section_keys.append('C#/Db')
        if k == 2:
            section_keys.append('D')
        if k == 3:
            section_keys.append('D#/Eb')
        if k == 4: 
            section_keys.append('E')
        if k == 5: 
            section_keys.append('F')
        if k == 6:
            section_keys.append('F#/Gb')
        if k == 7:
            section_keys.append('G')
        if k == 8:
            section_keys.append('G#/Ab')
        if k == 9:
            section_keys.append('A')
        if k == 10:
            section_keys.append('A#/Bb')
        if k == 11: 
            section_keys.append('B')
    return section_keys


# mode of key in section (tonal progression)    
def get_harmonic_mode_progression(song):   
    section_key_modes = []
    for section in song['sections']:
        m = section['mode']
        if m == -1:
            section_key_modes.append('Mode not Defined')
        if m == 0:
            section_key_modes.append('Minor')
        if m == 1:
            section_key_modes.append('Major')
    return section_key_modes


# tempo changes through out sections
def get_tempo_progression(song):    
    section_tempo = []
    for section in song['sections']:
        section_tempo.append(section['tempo'])
    return section_tempo


# returns section info on track passed through
def get_section_info(songs):
    section_list = []
    for song in tqdm(songs):
        h_prog = get_harmonic_progression(song)
        m_prog = get_harmonic_mode_progression(song)
        t_prog = get_tempo_progression(song)
        section_prog = [h_prog, m_prog, t_prog]
        section_list.append(section_prog)
    return section_list


# retrieve all converted track key and key mode from playlist
def get_track_global_key_modes(genre_id):
    print('Getting Track Keys')
    track_keys = get_all_keys(genre_id)
    print('Getting Track Modes')
    track_modes = get_all_key_modes(genre_id)
    print('Compiling Global Data')
    print('\n')
    harmonic_info = ([track_modes, track_keys])
    return harmonic_info

##############################################
# Function to Output all Engineered Features #
##############################################


# calls spotify playlist, returns section info on each track in playlist
def fetch_sections_info(genre_id):
    print('Getting Section Data')
    print('\n')
    data = call_spotify(genre_id)
    ids = fetch_ids(data)
    links = audio_analysis_links(ids)
    songs = fetch_analysis(links)
    section_info = get_section_info(songs)
    return section_info

# combine global and section info of each track from playlist 
# NOTE: for this function to work -- it needs a pivot table - or make function the cell below
def get_global_and_section_info(genre_id):
    print('Getting Global Data')
    print('\n')
    global_section_info = []
    global_info = get_track_global_key_modes(genre_id)
    print('Global Data Retrieved')
    print('\n')
    print('Getting Section Data')
    section_info = fetch_sections_info(genre_id)
    print('Section Data Retrieved')
    global_section_info = [global_info, section_info]
    print('Global and Section Data Compiled')
    return global_section_info

