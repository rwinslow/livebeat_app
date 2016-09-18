import json
import os
import re
import sys
import warnings

from flask import render_template
from flask import request
from flaskexample import app

from PIL import Image, ImageFilter
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

def scene_detection():
    """Train scene detection classifier"""

    # Print status
    print('Training scene detection classifier')

    # Build training set of classified images
    # basepath = '/Volumes/Passport/LiveBeat/'
    basepath = '/Users/Rich/Documents/Twitch'
    positive_path = os.path.join(basepath, 'test_images_button')
    negative_path = os.path.join(basepath, 'test_images_non-button')

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

def segmenter(video_id, model, pca, threshold=0.5, seconds_between_frames=60):
    """Generate timecodes for game and non-game segments"""

    # Print status
    print('Finding timecodes for segments')

    # Open file handle
    # basepath = '/Volumes/Passport/LiveBeat/'
    basepath = '/Users/Rich/Documents/Twitch'
    video_path = os.path.join(
        basepath,
        'video',
        'dota2ti_v{}_720p30.mp4'.format(video_id)
    )

    vid = imageio.get_reader(video_path, 'ffmpeg')

    # Get metadata and select 1 frame every n seconds
    meta = vid.get_meta_data()
    fps = int(np.round(meta['fps']))
    nframes = meta['nframes']
    frames = np.arange(0, nframes, seconds_between_frames*fps)

    # Check frames
    timecodes = []
    start_time = end_time = 0

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

        if prediction >= threshold and not start_time:
            start_time = '{}'.format(i)
        if prediction <= threshold and start_time:
            end_time = '{}'.format(i)
            timecodes.append(','.join([start_time, end_time]))
            print('Found:', [start_time, end_time])
            start_time = 0
            end_time = 0

    # Close video handle to release thread and buffer
    vid.close()

    # Process timecodes and flip switches for found game segments
    values = [0] * int(nframes/fps)
    df = pd.DataFrame(values)
    df.columns = ['game']
    for row in timecodes:
        # Filter out short segments of 7200 frames (4 * 60 fps * 30 sec)
        if stop - start > 7200:
            start, stop = (int(row[0])/fps, int(row[1])/fps)
            df.loc[(df.index >= start) & (df.index < stop), 'game'] = 1

    return df

def query_to_video(query):
    video_id = re.findall('\/(\d+)', query)[0]
    return video_id

def compile_chat(v_id):
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

    # basepath = '/Volumes/Passport/LiveBeat/'
    basepath = '/Users/Rich/Documents/Twitch'
    chat_path = os.path.join(basepath,
                             'chat',
                             'v{}'.format(v_id)
                            )

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
    df['secondstamp'] = df['timestamp'].apply(lambda x: int(round(x/1000)))

    # Create chat frequency data frame where index is no. of seconds into video
    chat_freq = pd.DataFrame(df['secondstamp'].value_counts().sort_index())
    chat_freq.columns = ['frequency']

    return chat_freq

def TEMP_HAVE_STAMPS(video_id):
    # Open file handle
    # basepath = '/Volumes/Passport/LiveBeat/'
    basepath = '/Users/Rich/Documents/Twitch'
    video_path = os.path.join(
        basepath,
        'video',
        'dota2ti_v{}_720p30.mp4'.format(video_id)
    )

    vid = imageio.get_reader(video_path, 'ffmpeg')

    # Get metadata and select 1 frame every n seconds
    meta = vid.get_meta_data()
    fps = int(np.round(meta['fps']))
    nframes = meta['nframes']
    frames = np.arange(0, nframes, 30*fps)

    vid.close()

    timecodes = [[36000, 39600],
    [82800, 90000],
    [241200, 244800],
    [277200, 453600],
    [475200, 478800],
    [532800, 734400],
    [788400, 792000],
    [799200, 802800],
    [896400, 1036800],
    [1044000, 1047600],
    [1134000, 1220400],
    [1231200, 1234800],
    [1285200, 1288800],
    [1299600, 1303200],
    [1335600, 1339200],
    [1389600, 1573200],
    [1677600, 1854000],]

    # Process timecodes and flip switches for found game segments
    values = [0] * int(nframes/fps)
    df = pd.DataFrame(values)
    df.columns = ['game']
    for row in timecodes:
        # Filter out short segments of 7200 frames (4 * 60 fps * 30 sec)
        if stop - start > 7200:
            start, stop = (row[0]/fps, row[1]/fps)
            df.loc[(df.index >= start) & (df.index < stop), 'game'] = 1

    return df

def build_scrub_plot(video_id, game, chat):
    fig, ax1 = plt.subplots(figsize=(18,2))
    ax2 = ax1.twinx()
    ax1.plot(game.index, game['game'], '.b', markersize=8)
    ax2.plot(chat.index, chat['frequency'], '-r')
    ax1.set_xlim([0, game.index.max()])

    basepath = '/Users/Rich/Documents/Flask/flaskexample/static/graphs'
    plt.savefig(
        os.path.join(
            basepath,
            '{}.png'.format(video_id)
        ),
        bbox_inches='tight',
    )

@app.route('/')
@app.route('/index')
def index():
    return render_template('master.html')

@app.route('/go', methods=['POST'])
def go():
    # Process query to identify video of request
    query = request.form['query']
    video_id = query_to_video(query)

    # Get scene detector and acquire game timecodes
    # model, pca = scene_detection()
    # game = segmenter(video_id, model, pca)
    game = TEMP_HAVE_STAMPS(video_id)

    # Get chat data
    chat = compile_chat(video_id)

    # Build video path
    # basepath = '/Volumes/Passport/LiveBeat/'
    basepath = '/Users/Rich/Documents/Twitch'
    video_path = os.path.join(
        basepath,
        'video',
        'dota2ti_v{}_720p30.mp4'.format(video_id)
    )

    # Build image for scrub plot
    build_scrub_plot(video_id, game, chat)

    return render_template(
        'go.html',
        query = query,
        video_id = video_id,
    )