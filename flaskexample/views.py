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

    # # Process status results and filter out short sections
    # for i in range(1, len(statuses)-3):
    #     if (statuses[i][3] >= 1
    #         and statuses[i-1][3] == 0
    #         and statuses[i+2][3] == 0):
    #         statuses[i][1:] = (0, 0, 0)
    #         statuses[i+1][1:] = (0, 0, 0)

    status = pd.DataFrame(statuses)
    status.columns = ['second', 'game']

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

    # Aggregate files into dictionary
    aggregated = []
    for f in file_list(chat_path):
        get_path = os.path.join(chat_path, f)
        with open(get_path) as c:

            # Format line and separate multiple JSON strings with commas
            line = '[{}]'.format(c.readline()).replace('}}{', '}},{')
            data = json.loads(line)[0]

            for message in data['data']:
                aggregated.append(agg_chat(message))

    # Build data frame from chat results
    df = pd.DataFrame(aggregated)
    minimum = df['timestamp'].min()
    maximum = df['timestamp'].max()
    df['timestamp'] = df['timestamp'].apply(lambda x: x - minimum)
    df['secondstamp'] = df['timestamp'].apply(
        lambda x: int(round(x/1000/seconds_per_bin)*seconds_per_bin)
    )

    return df


@app.route('/')
@app.route('/index')
def index():
    return render_template('master.html')

@app.route('/test')
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

    # Get chat data
    chat = compile_chat(chat_path)

    # Get scene detector and acquire game timecodes
    positive_path = os.path.join(basepath, 'test_images_button')
    negative_path = os.path.join(basepath, 'test_images_non-button')
    model, pca = scene_detection(positive_path, negative_path)

    status = segmenter(video_path, model, pca)

    # Extract features to generate graphs
    graph_x = ','.join(status['second'].values.astype(str).tolist())
    graph_game = ','.join(status['game'].values.astype(str).tolist())
    graph_chat = ','.join(chat['frequency'].values.astype(str).tolist())

    return render_template(
        'go.html',
        query = query,
        video_id = video_id,
        graph_x = graph_x,
        graph_red = graph_game,
        graph_chat = graph_chat
    )
