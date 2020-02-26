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
                     '0hLKYU2Wwv5WTu7XwnesPg',
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


#################################
# Feature Engineer Function 1.2 #
#################################

#Now that the dataframe is generated. There are a few more steps necessary before this data can be used.

# 1. Clean dataframe cells so each cell has strings and not lists
# 2. Transform the Harmonic Progression into Roman Numeral Analysis

# function to convert list of words into string
def l_to_s(_list):
    s = ''
    for i in _list:
        s = s + i + ','
    return s[:-2] 

# function to convert list of integers into a string
def l_to_int(_list):
    s = ''
    for i in _list:
        s = s + str(i) + ','
    return s[:-2] 

########################################################
# Make a DataFrame that has All Roman Numeral Analysis #
########################################################

from harmonic_function import *

def get_all_global_modes(dataframe):
    global_modes = []
    for i in dataframe['Track Modes']:
        global_modes.append(i)
    return global_modes

def get_all_global_keys(dataframe):
    global_keys = []
    for i in dataframe['Track Keys']:
        global_keys.append(i)
    return global_keys
        
def get_all_harmonic_prog(dataframe):
    harmonic_prog = []
    for i in dataframe['Harmonic Progression']:
        harmonic_prog.append(i)
    return harmonic_prog


def retrieve_roman_numeral_analysis(dataframe):
    modes = get_all_global_modes(dataframe)
    keys = get_all_global_keys(dataframe)
    harmonic_prog = get_all_harmonic_prog(dataframe)
    analysis = []
    for i in list(range(len(dataframe))):
        analysis.append(get_progression_m(modes[i], keys[i], harmonic_prog[i]))
    return analysis


# finding ways to count chords and define a roman numeral analysis's underlaying harmonic structure

def get_harmonic_signature(analysis):
    duplicate_list = []
    unique_list = []
    for i in analysis:
        if analysis.count(i) > 1:  
            duplicate_list.append(i)
        if analysis.count(i) == 1:
            unique_list.append(i)
    print('Sonorus Chords')
    print(set(duplicate_list))
    print('Coloring Chords')
    print(unique_list)
    print('\n')
    Harmonic_signature = [set(duplicate_list), unique_list]
    print('Harmonic Signature')
    return Harmonic_signature    


######################
# HARMONIC SIGNATURE #
######################

import re



# function that takes in list of analysis and out puts a list of two lists
# first list is cleaned and accurate roman numerial anaylsis
# second list is a numeric conversion to help reduce harmonic categories
def get_numeric_conversion(list_of_analysis):
    major_chord_analysis_finder = re.compile('^M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$')
    minor_i_chord_analysis_finder = re.compile('^i$')
    minor_v_chord_analysis_finder = re.compile('^v$')
    minor_ii_iv_chord_analysis_finder = re.compile('^i.$')
    minor_iii_chord_analysis_finder = re.compile('^iii$')
    minor_vi_chord_analysis_finder = re.compile('^vi$')
    minor_vii_chord_analysis_finder = re.compile('^vii$')
    minor_dim_vii_chord_analysis_finder = re.compile('^vii..$')
    sharp_chord_analysis_finder = re.compile('^#..$')
    sharp_chord_analysis_finder_ = re.compile('^#...$')
    flat_chord_analysis_finder = re.compile('^b..$')
    flat_chord_analysis_finder_ = re.compile('^b...$')
    new_list_of_analysis = [[] for l in range(len(list_of_analysis))]
    numerical_conversion = [[] for l in range(len(list_of_analysis))]
    count = 0
    for i in list_of_analysis:
        for a in i:
            print(count)
            major_result = re.findall(major_chord_analysis_finder, a)
            minor_i_result = re.findall(minor_i_chord_analysis_finder, a)
            minor_v_result = re.findall(minor_v_chord_analysis_finder, a)
            minor_ii_iv_result = re.findall(minor_ii_iv_chord_analysis_finder, a)
            minor_iii_result = re.findall(minor_iii_chord_analysis_finder, a)
            minor_vi_result = re.findall(minor_vi_chord_analysis_finder, a)
            minor_vii_result = re.findall(minor_vii_chord_analysis_finder, a)
            minor_dim_vii_result = re.findall(minor_dim_vii_chord_analysis_finder, a)
            sharp_result = re.findall(sharp_chord_analysis_finder, a)
            sharp_result_ = re.findall(sharp_chord_analysis_finder_, a)
            flat_result = re.findall(flat_chord_analysis_finder, a)
            flat_result_ = re.findall(flat_chord_analysis_finder_, a)
            if major_result:
                print("Major Chords found.")
                print(major_result)
                new_list_of_analysis[count].append(major_result)
                numerical_conversion[count].append('1')
            if minor_i_result:
                print("Minor Chords found.")	
                print(minor_i_result)
                new_list_of_analysis[count].append(minor_i_result)
                numerical_conversion[count].append('0')
            if minor_v_result:
                print(minor_v_result)
                new_list_of_analysis[count].append(minor_v_result)
                numerical_conversion[count].append('0')
            if minor_ii_iv_result:
                print(minor_ii_iv_result)
                new_list_of_analysis[count].append(minor_ii_iv_result)
                numerical_conversion[count].append('0')
            if minor_iii_result:
                print(minor_iii_result)
                new_list_of_analysis[count].append(minor_iii_result)
                numerical_conversion[count].append('0')
            if minor_vi_result:
                print(minor_vi_result)
                new_list_of_analysis[count].append(minor_vi_result)
                numerical_conversion[count].append('0')
            if minor_vii_result:
                print(minor_vii_result)
                new_list_of_analysis[count].append(minor_vii_result)
                numerical_conversion[count].append('0')
            if minor_dim_vii_result:
                print(minor_dim_vii_result)
                new_list_of_analysis[count].append(minor_dim_vii_result)
                numerical_conversion[count].append('0')
            if sharp_result:
                print("Sharp Chords Found")
                print(sharp_result)
                new_list_of_analysis[count].append(sharp_result)
                numerical_conversion[count].append('#')
            if sharp_result_:
                print(sharp_result_)
                new_list_of_analysis[count].append(sharp_result_)
                numerical_conversion[count].append('#')
            if flat_result:
                print('Flat Chords Found')
                print(flat_result)
                new_list_of_analysis[count].append(flat_result)
                numerical_conversion[count].append('b')
            if flat_result_:
                print(flat_result_)
                new_list_of_analysis[count].append(flat_result_)
                numerical_conversion[count].append('b')
            else:
                pass
        count += 1
    cleaned_list = [[] for l in range(len(new_list_of_analysis))]
    count = 0
    for x in new_list_of_analysis:
        for i in x:
            for a in i:
                if type(a) == tuple:
                    x = [(tuple(int(x) if x.isdigit() else x for x in _ if x)) for _ in i]
                    x = ''.join(x[0])
                    cleaned_list[count].append(x)
                if type(a) == str:
                    cleaned_list[count].append(a)
        count += 1
    return cleaned_list, numerical_conversion



# combining the full encoding process into one function
def encode_shape_and_color(converted):
    harmonic_signatures = []
    for i in converted[1]: 
        x = get_harmonic_signature(i)
        harmonic_signatures.append(x)
        harmonic_sig_df = pd.DataFrame(harmonic_signatures)
        harmonic_sig_df = harmonic_sig_df.rename(columns={0: 'Shape', 1: 'Color'})
    harmonic_sig_df['Shape'] = harmonic_sig_df['Shape'].apply(lambda x: str(list(x)))
    harmonic_sig_df['Color'] = harmonic_sig_df['Color'].apply(lambda x: str(list(x)))
    encoded_shape = pd.get_dummies(harmonic_sig_df['Shape'], prefix = 'Shape')
    encoded_color = pd.get_dummies(harmonic_sig_df['Color'], prefix = 'Color')
    # encoded_shape.columns = ['Shape_0', 
    #                          'Shape_1', 
    #                          'Shape_2', 
    #                          'Shape_3', 
    #                          'Shape_4',
    #                          'Shape_5',
    #                          'Shape_6',
    #                          'Shape_7']
    # encoded_color.columns = ['Color_0',
    #                          'Color_1',
    #                          'Color_2',
    #                          'Color_3',
    #                          'Color_4',
    #                          'Color_5',
    #                          'Color_6']
    shape_and_color_df = pd.concat([encoded_shape, 
                                    encoded_color], 
                                    axis = 1)
    return shape_and_color_df