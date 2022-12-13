from config import Config
import os
import glob
import cv2
import numpy as np
import tensorflow_hub as hub

class StyleTransfer:
    def __init__(self, config=Config):
        self.config = config
        self.hub_module = hub.load(self.config.TENSORFLOW_HUB_HANDLE)
        self.pre_frame_dir = glob.glob(f'{self.config.PRE_VID_FRAME_DIR}/*')
        self.post_frame_dir = glob.glob(f'{self.config.POST_VID_FRAME_DIR}/*')
        self.style_dir = glob.glob(f'{self.config.STYLE_DIR}/*')
        self.ref_img_count = len(self.config.STYLE_SEQ)

        delete_these_files = self.post_frame_dir
        if self.config.CLEAR_INPUT_FRAME_CACHE:
            delete_these_files += self.pre_frame_dir
        
        for file in delete_these_files:
            os.remove(file)

         # Update contents of directory after deletion
        self.pre_frame_dir = glob.glob(f'{self.config.PRE_VID_FRAME_DIR}/*')
        self.post_frame_dir = glob.glob(f'{self.config.POST_VID_FRAME_DIR}/*')

        if len(self.pre_frame_dir) > 0:
            self.frame_width = cv2.imread(self.pre_frame_dir[0].shape[1])

    def fetch_pre_frames(self):
        if len(self.pre_frame_dir) > 0:
            print("Using cached frames")
            return
        
        video = cv2.VideoCapture(self.config.PRE_VID_PATH)
        frame_interval = np.floor((1.0 / self.config.FPS) * 1000)

        ret,frame = video.read()
        
        if frame is None:
                raise ValueError(f"Error: No video provided")
    
        ## Adjust scale based on specified frame height
        scale = self.conf.FRAME_HEIGHT / frame.shape[0]
        self.frame_width = int(frame.shape[1] * scale)
        
        frame = cv2.resize(frame, (self.frame_width, self.conf.FRAME_HEIGHT)).astype(np.uint8)
        cv2.imwrite(self.conf.PRE_VID_FRAME_PATH.format(0), frame)
        
        ## Sample original video at specified frame rate
        offset = 1
        while ret:
            timestamp = offset * frame_interval
            video.set(cv2.CAP_PROP_POS_MSEC, timestamp)
            ret, frame = video.read()
            if not ret:
                break
            frame = cv2.resize(frame, (self.frame_width, self.conf.FRAME_HEIGHT)).astype(np.uint8)
            cv2.imwrite(self.conf.PRE_VID_FRAME_PATH.format(0), frame)
            offset += 1

        self.pre_frame_dir = glob.glob(f'{self.conf.PRE_VID_FRAME_DIR}/*')


    def fetch_style_refs(self):
        frame_len = len(self.pre_frame_dir)
        style_ref_imgs = list()
        style_ref_img_resized = False
        style_ref_img_files = sorted(self.style_dir)
        # update t const if poss
        self.t_const = frame_len if self.ref_img_count == 1 else np.ceil(frame_len / (self.ref_img_count - 1))
        self.transition_style_seq_list = list()

        # Make all style ref imgs same size as first style
        style_ref_img_1_height, style_ref_img_1_width = None

        for style_ref_img_file in style_ref_img_files:
            style_ref_img = cv2.imread(style_ref_img_file)
            style_ref_img = cv2.cvtColor(style_ref_img, cv2.COLOR_BGR2RGB)
            if style_ref_img_1_height is None or style_ref_img_1_width is None:
                style_ref_img_1_height, style_ref_img_1_width, channels = style_ref_img.shape
            else:
                style_ref_img_height, style_ref_img_width, channels = style_ref_img.shape
                # Change these style imgs to match first style img
                if style_ref_img_1_height != style_ref_img_height or style_ref_img_1_width != style_ref_img_width:
                    style_ref_img = cv2.resize(style_ref_img, (style_ref_img_1_width, style_ref_img_1_height))
                    style_ref_img_resized = True
            style_ref_imgs.append(style_ref_img / self.MAX_CHANNEL_INTENSITY)
            
        # Alert user that style images were resized
        if style_ref_img_resized:
            print("Warning: Resizing style images -> may cause distortion")
        
        for i in range(self.ref_img_count):
            style_seq_num = self.config.STYLE_SEQ[i]
            if style_seq_num is None:
                self.transition_style_seq_list.append(None)
            else:
                self.transition_style_seq_list.append(style_ref_imgs[style_seq_num])
                
    
    def run(self):
        print("Fetching input frames")
        self.fetch_pre_frames()
        print("Fetching style reference info")
        self.fetch_style_refs()
        print("Fetching output frames")
        self.fetch_post_frames()
        print("Saving video")
        self.generate_stylized_video()

## Run as a script
if __name__ == "__main__":
    StyleTransfer().run()