class Config:
    
    ROOT = '.'
    # defines the maximum height dimension in pixels. Used for down-sampling the video frames
    FRAME_HEIGHT = 360 ## fine tune parameter
    
    CLEAR_INPUT_FRAME_CACHE = False
    
    # defines the rate at which you want to capture frames from the input video
    FPS = 20 ## fine tune parameter

    # input video variables
    PRE_VID_NAME = 'pre_vid.mov'
    PRE_VID_PATH = f'{ROOT}/{PRE_VID_NAME}'

    # input video frame variables
    PRE_VID_FRAME_DIR = f'{ROOT}/pre_frames'
    PRE_VID_FRAME_FILE = 'frame_{:0>4d}_.png'
    PRE_VID_FRAME_PATH = f'{PRE_VID_FRAME_DIR}/{PRE_VID_FRAME_FILE}'

    STYLE_DIR = f'{ROOT}/style_ref'
    STYLE_IMG_FILE = 'style_{:0>4d}_.png'
    STYLE_IMG_PATH= f'{STYLE_DIR}/{STYLE_IMG_FILE}'

    # defines the reference style image transition sequence. Values correspond to indices in STYLE_REF_DIRECTORY
    # add None in the sequence to NOT apply style transfer for part of the video (ie. [None, 0, 1, 2])  

    # output video variables
    POST_VID_NAME = 'post_video.mp4'
    POST_VID_PATH = f'{ROOT}/{POST_VID_NAME}'

    # output video frame variables
    POST_VID_FRAME_DIR = f'{ROOT}/post_frames'
    POST_VID_FRAME_FILE = 'frame_{:0>4d}_.png'
    POST_VID_FRAME_PATH = f'{POST_VID_FRAME_DIR}/{POST_VID_FRAME_FILE}'

    GHOST_FRAME_TRANSPARENCY = 0.5 ## fine tune parameter
    KEEP_COLORS = False

    TENSORFLOW_CACHE_DIRECTORY = f'{ROOT}/tensorflow_cache'
    TENSORFLOW_HUB_HANDLE = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'