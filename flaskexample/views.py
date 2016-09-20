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

def get_health_status(img, **kwargs):
    # Player status
    h, w, c = img.shape
    factor = 0.226
    y1 = int(h * 0.044)
    x1 = int(w * (0.5 - factor))
    y2 = y1+1
    x2 = int(w * (0.5 + factor))
    pl_status = img[y1:y2, x1:x2, :]

    # Block out center of player status
    h, w, c = pl_status.shape
    factor = .12
    x1 = int(np.round(w * (0.5 - factor)))+1
    x2 = int(np.round(w * (0.5 + factor)))
    pl_status[:, x1:x2, :] = 0

    pl_status = pl_status[:,:,::-1] # Put player status from RGB to BGR for OpenCV
    hsv = cv2.cvtColor(pl_status,cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0], None, [256], [0,256])
    s = cv2.calcHist([hsv], [1], None, [256], [0,256])
    v = cv2.calcHist([hsv], [2], None, [256], [0,256])

    # Initialize colors
    red = int(sum(h[1:15]))
    green = int(sum(h[50:65]))
    if red == 0 or green == 0:
        return (0, 0)
    # else:
    #     # Save image for validation
    #     basepath = '/Volumes/Passport/LiveBeat/'
    #     save_path = os.path.join(basepath, 'validation', '{}.png'.format(i))
    #     Image.fromarray(pl_status).save(save_path)
    return (red, green)

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

        # Get player status
        red, green = get_health_status(img, i=i)

        # Generate predictions for each selected frame
        features = pca.transform(img2features(shop))
        features = np.array(features).reshape(1, -1)
        prediction = model.predict(features)

        second = int(i/fps)
        if prediction >= threshold:
            statuses.append([second, red, green, 1])
        else:
            statuses.append([second, 0, 0, 0])

    # Close video handle to release thread and buffer
    vid.close()

    # Process status results and filter out short sections
    for i in range(1, len(statuses)-3):
        if (statuses[i][3] == 1
            and statuses[i-1][3] == 0
            and statuses[i+2][3] == 0):
            statuses[i][1:] = (0, 0, 0)
            statuses[i+1][1:] = (0, 0, 0)

    status = pd.DataFrame(statuses)
    status.columns = ['second', 'red', 'green', 'game']

    # status.to_csv('/Users/Rich/Documents/Twitch/statuses/color_hist_check.csv')
    #
    # red_threshold = status['red'].mean() + 3*status['red'].std()
    # status.loc[
    #         status['red'] > red_threshold, 'red'
    # ] = status['red'].median()
    #
    # green_threshold = status['green'].mean() + 3*status['green'].std()
    # status.loc[
    #         status['green'] > red_threshold, 'green'
    # ] = status['green'].median()

    red_max = status['red'].max()
    green_max = status['green'].max()
    status['red'] = status['red'].apply(
            lambda x: x/red_max
    ).round(2)
    status['green'] = status['green'].apply(
            lambda x: -1*x/green_max
    ).round(2)

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

    # Create chat frequency data frame where index is no. of seconds into video
    chat_freq = pd.DataFrame(df['secondstamp'].value_counts().sort_index())
    chat_freq.columns = ['frequency']

    # Normalize frequency for plotting
    _max = chat_freq['frequency'].max()
    _min = chat_freq['frequency'].min()
    chat_freq['frequency'] = chat_freq['frequency'].apply(
        lambda x: (x - _min) / (_max - _min)
    )

    chat_freq.to_csv('/Users/Rich/Documents/Twitch/chat_scale/test.csv')

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
        start, stop = (row[0]/fps, row[1]/fps)
        # Filter out short segments of 120 seconds (4 * 30 sec)
        if stop - start > 120:
            df.loc[(df.index >= start) & (df.index < stop), 'game'] = 1

    return df

def build_scrub_plot(target_path, game, chat):
    fig, ax1 = plt.subplots(figsize=(18,2))
    ax2 = ax1.twinx()
    ax1.plot(game.index, game['game'], '.b', markersize=8)
    ax2.plot(chat.index, chat['frequency'], '-r')
    ax1.set_xlim([0, game.index.max()])

    plt.savefig(
        target_path,
        bbox_inches='tight',
    )

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
    # status.to_csv('/Users/Rich/Documents/Twitch/statuses/test_value.csv')
    # game = TEMP_HAVE_STAMPS(video_id)

    # Build image for scrub plot
    # basepath = '/Users/Rich/Documents/Flask/flaskexample/static/graphs'
    # target_path = os.path.join(basepath, '{}.png'.format(video_id))

    # Extract features to generate graphs
    graph_x = ','.join(status['second'].values.astype(str).tolist())
    graph_red = ','.join(status['red'].values.astype(str).tolist())
    graph_green = ','.join(status['green'].values.astype(str).tolist())
    graph_chat = ','.join(chat['frequency'].values.astype(str).tolist())

    return render_template(
        'go.html',
        query = query,
        video_id = video_id,
        graph_x = graph_x,
        graph_red = graph_red,
        graph_green = graph_green,
        graph_chat = graph_chat
    )
