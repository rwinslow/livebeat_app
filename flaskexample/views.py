import json
import os
import re
import sys
import warnings

from flask import render_template
from flask import request
from flaskexample import app

from PIL import Image, ImageFilter
import cv2
import imageio
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn import decomposition
from sklearn.preprocessing import PolynomialFeatures, normalize
from sklearn.linear_model import LogisticRegression
from scipy.misc import imresize
from skimage import feature, color

# Hide scikit-learn warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def file_list(start_dir):
    """Generate file list in directory"""
    file_list = []
    for root, dirs, files in os.walk(start_dir):
        for f in files:
            if f[0] != '.':
                file_list.append(f)
    return file_list

def hog_img(img):
    """Generate Histogram of Oriented Gradients for image"""
    # img = np.asarray(img)
    hogged, hogged_img = feature.hog(color.rgb2gray(img), visualise=True)
    return hogged

def img2features(img):
    return hog_img(img)

def scene_detection(positive_path, negative_path):
    """Train scene detection classifier"""

    # Print status
    print('Training scene detection classifier')

    # Build training set of classified images
    positive_list = file_list(positive_path)
    negative_list = file_list(negative_path)
    X = []
    y = []

    for f in positive_list:
        img = np.array(Image.open(os.path.join(positive_path, f)))
        X.append(img2features(img))
        y.append(1)

    for f in negative_list:
        img = np.array(Image.open(os.path.join(negative_path, f)))
        X.append(img2features(img))
        y.append(0)

    # PCA to get valuable features
    pca = decomposition.PCA(n_components=4)
    pca.fit(X)
    X = pca.transform(X)

    # Train model
    model = LogisticRegression()
    model.fit(X, y)

    return (model, pca)

def segmenter(video_path, model, pca, threshold=0.5, seconds_between_frames=60):
    """Generate timecodes for game and non-game segments"""

    # Print status
    print('Finding timecodes for segments')

    # Open file handle
    vid = imageio.get_reader(video_path, 'ffmpeg')

    # Get metadata and select 1 frame every n seconds
    meta = vid.get_meta_data()
    fps = int(np.round(meta['fps']))
    nframes = meta['nframes']
    frames = np.arange(0, nframes, seconds_between_frames*fps)

    # Output details to console
    print('Video:', video_path)
    print('Frames:', nframes)
    print('FPS:', fps)

    # Check frames
    timecodes = []
    statuses = []
    start_time = end_time = 0
    segment = 1

    # Run through frames and find segments
    for i in frames:
        img = vid.get_data(i)

        # Resize image appropriately for training set
        img = imresize(img, (720, 1280))

        # Isolate shop button
        h, w, c = img.shape
        x1 = int(w * .94)
        x2 = x1 + 76
        y1 = int(h * .814)
        y2 = y1 + 26
        shop = img[y1:y2, x1:x2, :]

        # Generate predictions for each selected frame
        features = pca.transform(img2features(shop))
        features = np.array(features).reshape(1, -1)
        prediction = model.predict(features)

        second = int(i/fps)
        if prediction >= threshold:
            statuses.append([second, segment])
        else:
            segment += 1
            statuses.append([second, 0])

    # Close video handle to release thread and buffer
    vid.close()

    status = pd.DataFrame(statuses)
    status.columns = ['second', 'game']

    # Filter out isolated bips of game time
    counts = pd.DataFrame(status['game'].value_counts())
    idx = counts.loc[counts['game'] <= 2].index.tolist()
    status.loc[status['game'].isin(idx), 'game'] = 0

    return status

def query_to_video(query):
    video_id = re.findall('\/(\d+)', query)[0]
    return video_id

def compile_chat(chat_path, seconds_per_bin=60):
    def agg_chat(data):
        """Aggregate JSON chat data into a row"""
        attr = data['attributes']

        timestamp = attr['timestamp']
        message = attr['message']
        author = attr['from']
        turbo = attr['tags']['turbo']
        sub = attr['tags']['subscriber']

        try:
            emotes = attr['tags']['emotes']
            emote_count = sum([len(emotes[key]) for key in emotes.keys()])
        except:
            emote_count = 0

        row = {
            'timestamp': timestamp,
            'author': author,
            'message': message,
            'turbo': turbo,
            'sub': sub,
            'emote_count': emote_count
        }

        return row

    def agg_emotes(data):
        """Aggregate JSON chat emotes into a row"""
        attr = data['attributes']

        timestamp = attr['timestamp']

        try:
            emotes = attr['tags']['emotes']
            rows = []
            for key in emotes.keys():
                rows.append(
                    {
                        'timestamp': timestamp,
                        'emote': key,
                        'count': len(emotes[key])
                    }
                )
            return rows
        except AttributeError:
            return [{'timestamp': timestamp,
                    'emote': 0,
                    'count': 0
                   }]

    # Aggregate files into dictionary
    aggregate_chat = []
    aggregate_emotes = []
    for f in file_list(chat_path):
        get_path = os.path.join(chat_path, f)
        with open(get_path) as c:

            # Format line and separate multiple JSON strings with commas
            line = '[{}]'.format(c.readline()).replace('}}{', '}},{')
            data = json.loads(line)[0]

            for message in data['data']:
                aggregate_chat.append(agg_chat(message))
                aggregate_emotes.extend(agg_emotes(message))

    # Build data frame from chat results
    df_chat = pd.DataFrame(aggregate_chat)
    minimum = df_chat['timestamp'].min()
    maximum = df_chat['timestamp'].max()
    df_chat['timestamp'] = df_chat['timestamp'].apply(lambda x: x - minimum)
    df_chat['secondstamp'] = df_chat['timestamp'].apply(
        lambda x: int(round(x/1000/seconds_per_bin)*seconds_per_bin)
    )

    # Build data frame from chat results
    df_emotes = pd.DataFrame(aggregate_emotes)
    minimum = df_emotes['timestamp'].min()
    maximum = df_emotes['timestamp'].max()
    df_emotes['timestamp'] = df_emotes['timestamp'].apply(lambda x: x - minimum)
    df_emotes['secondstamp'] = df_emotes['timestamp'].apply(
        lambda x: int(round(x/1000/seconds_per_bin)*seconds_per_bin)
    )

    return df_chat, df_emotes

def get_highlights(status, chat):
    games = status['game'].value_counts().index.tolist()[1:]
    highlights = pd.DataFrame()
    highlights['second'] = status['second']
    highlights['highlight'] = [0] * len(highlights.index)
    for game in games:
        seconds = status[status['game'] == game]['second']
        start = seconds.min()
        stop = seconds.max()

        chat_segment = chat[
            (chat['secondstamp'] >= start) & (chat['secondstamp'] <= stop)
        ]
        game_highs = chat_segment['secondstamp'].value_counts(
            ).sort_values(ascending=False).index[0:3].tolist()
        for i in game_highs:
            highlights.loc[highlights['second'] == i, 'highlight'] = 1

    return highlights

@app.route('/')
@app.route('/index')
def index():
    return render_template('master.html')

@app.route('/test', methods=['POST'])
def test():
    return render_template('overlay_test.html')

@app.route('/go', methods=['POST'])
def go():
    # Define basepath for file locations
    basepath = '/Volumes/Passport/LiveBeat/'
    # basepath = '/Users/Rich/Documents/Twitch'

    # Process query to identify video of request
    query = request.form['query']
    video_id = query_to_video(query)
    video_file = 'dota2ti_v{}_720p30.mp4'.format(video_id)
    video_path = os.path.join(basepath, 'video', video_file)
    chat_path = os.path.join(basepath, 'chat', 'v{}'.format(video_id))

    # Get chat and emote data
    chat, emotes_deep = compile_chat(chat_path)

    # Get rid of rows without any emotes
    # emotes_deep to dig into frequency of specific emotes
    emotes_deep = emotes_deep[emotes_deep['emote'] != 0]

    # Segment chat by emotes
    emotes = chat.loc[
        chat['emote_count'] > 0, ['emote_count', 'secondstamp']
    ]
    no_emotes = chat.loc[
        chat['emote_count'] == 0, ['emote_count', 'secondstamp']
    ]

    #
    # PROCESS CHAT WITH EMOTES
    #
    emotes_values = pd.DataFrame(
        emotes['secondstamp'].value_counts().sort_index()
    )
    emotes_values.columns = ['frequency']

    # Normalize frequency for plotting
    _max = emotes_values['frequency'].max()
    _min = emotes_values['frequency'].min()
    emotes_values['frequency'] = emotes_values['frequency'].apply(
        lambda x: (x - _min) / (_max - _min)
    )

    emotes_list = emotes_values['frequency'].values.astype(str).tolist()

    #
    # PROCESS CHAT WITHOUT EMOTES
    #
    no_emotes_values = pd.DataFrame(
        no_emotes['secondstamp'].value_counts().sort_index()
    )
    no_emotes_values.columns = ['frequency']

    # Normalize frequency for plotting
    _max = no_emotes_values['frequency'].max()
    _min = no_emotes_values['frequency'].min()
    no_emotes_values['frequency'] = no_emotes_values['frequency'].apply(
        lambda x: (x - _min) / (_max - _min)
        )

    no_emotes_list = no_emotes_values['frequency'].values.astype(
        str).tolist()

    #
    # PROCESS CHAT MESSAGE LENGTH
    #
    chat['message_len'] = chat['message'].apply(lambda x: len(x))
    chat_mean_list = chat.groupby('secondstamp')['message_len'].mean(
        ).tolist()
    _max = max(chat_mean_list)
    _min = min(chat_mean_list)
    chat_mean_list = [
        str((v - _min)/(_max - _min)) for v in chat_mean_list
    ]

    # Get scene detector and acquire sections where games are
    positive_path = os.path.join(basepath, 'test_images_button')
    negative_path = os.path.join(basepath, 'test_images_non-button')
    model, pca = scene_detection(positive_path, negative_path)
    status = segmenter(video_path, model, pca)
    second_list = status['second'].values.astype(str).tolist()
    game_list = [
            '1' if v > 0 else '0' for v in status['game'].values.tolist()]

    # Identify highlights
    try:
        highlights = get_highlights(status, no_emotes)
        highlights_list = highlights['highlight'].values.astype(str).tolist()
    except:
        # If there's no chat data
        highlights_list = ['0']

    # Generate graph strings from lists
    graph_x = ','.join(second_list)
    graph_game = ','.join(game_list)
    graph_chat = ','.join(no_emotes_list)
    graph_highlights = ','.join(highlights_list)
    graph_chat_len = ','.join(chat_mean_list)
    graph_emote_chat = ','.join(emote_list)

    return render_template(
        'go.html',
        query = query,
        video_id = video_id,
        graph_x = graph_x,
        graph_game = graph_game,
        graph_chat = graph_chat,
        graph_highlights = graph_highlights,
        graph_chat_len = graph_chat_len,
        graph_emote_chat = graph_emote_chat,
    )
