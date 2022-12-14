from config import Config
import os
import glob
import cv2
import numpy as np
import tensorflow_hub as hub
import tensorflow as tf

class StyleTransfer:

    MAX_CHANNEL_INTENSITY = 255.0
    
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
        scale = self.config.FRAME_HEIGHT / frame.shape[0]
        self.frame_width = int(frame.shape[1] * scale)
        
        frame = cv2.resize(frame, (self.frame_width, self.config.FRAME_HEIGHT)).astype(np.uint8)
        cv2.imwrite(self.config.PRE_VID_FRAME_PATH.format(0), frame)
        
        ## Sample original video at specified frame rate
        offset = 1
        while ret:
            timestamp = offset * frame_interval
            video.set(cv2.CAP_PROP_POS_MSEC, timestamp)
            ret, frame = video.read()
            if not ret:
                break
            frame = cv2.resize(frame, (self.frame_width, self.config.FRAME_HEIGHT)).astype(np.uint8)
            cv2.imwrite(self.config.PRE_VID_FRAME_PATH.format(offset), frame)
            offset += 1

        self.pre_frame_dir = glob.glob(f'{self.config.PRE_VID_FRAME_DIR}/*')


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

    def _trim_img(self, img):
        return img[:self.config.FRAME_HEIGHT, :self.frame_width]

    def fetch_post_frames(self):
        self.pre_frame_dir = glob.glob(f'{self.config.PRE_VID_FRAME_DIR}/*')
        ghost_frame = None
        for count, filename in enumerate(sorted(self.pre_frame_dir)):
            if count % 10 == 0:
                print(f"Output frame: {(count/len(self.pre_frame_dir)):.0%}")
            content_img = cv2.imread(filename) 
            content_img = cv2.cvtColor(content_img, cv2.COLOR_BGR2RGB) / self.MAX_CHANNEL_INTENSITY
            curr_style_img_index = int(count / self.t_const)
            mix_ratio = 1 - ((count % self.t_const) / self.t_const)
            inv_mix_ratio = 1 - mix_ratio

            prev_image = self.transition_style_seq_list[curr_style_img_index]
            next_image = self.transition_style_seq_list[curr_style_img_index + 1]
            
            prev_is_content_img = False
            next_is_content_img = False
            if prev_image is None:
                prev_image = content_img
                prev_is_content_img = True
            if next_image is None:
                next_image = content_img
                next_is_content_img = True
            # If both, don't need to apply style transfer
            if prev_is_content_img and next_is_content_img:
                temp_ghost_frame = cv2.cvtColor(ghost_frame, cv2.COLOR_RGB2BGR) * self.MAX_CHANNEL_INTENSITY
                cv2.imwrite(self.config.POST_VID_FRAME_PATH.format(count), temp_ghost_frame)
                continue
            
            if count > 0:
                content_img = ((1 - self.config.GHOST_FRAME_TRANSPARENCY) * content_img) + (self.config.GHOST_FRAME_TRANSPARENCY * ghost_frame)
            content_img = tf.cast(tf.convert_to_tensor(content_img), tf.float32)

            if prev_is_content_img:
                blended_img = next_image
            elif next_is_content_img:
                blended_img = prev_image
            else:
                prev_style = mix_ratio * prev_image
                next_style = inv_mix_ratio * next_image
                blended_img = prev_style + next_style

            blended_img = tf.cast(tf.convert_to_tensor(blended_img), tf.float32)
            expanded_blended_img = tf.constant(tf.expand_dims(blended_img, axis=0))
            expanded_content_img = tf.constant(tf.expand_dims(content_img, axis=0))
            # Apply style transfer
            stylized_img = self.hub_module(expanded_content_img, expanded_blended_img).pop()
            stylized_img = tf.squeeze(stylized_img)

            # Re-blend
            if prev_is_content_img:
                prev_style = mix_ratio * content_img
                next_style = inv_mix_ratio * stylized_img
            if next_is_content_img:
                prev_style = mix_ratio * stylized_img
                next_style = inv_mix_ratio * content_img
            if prev_is_content_img or next_is_content_img:
                stylized_img = self._trim_img(prev_style) + self._trim_img(next_style)

            if self.config.KEEP_COLORS:
                stylized_img = self._color_correct_to_input(content_img, stylized_img)
            
            ghost_frame = np.asarray(self._trim_img(stylized_img))

            temp_ghost_frame = cv2.cvtColor(ghost_frame, cv2.COLOR_RGB2BGR) * self.MAX_CHANNEL_INTENSITY
            cv2.imwrite(self.config.POST_VID_FRAME_PATH.format(count), temp_ghost_frame)
        self.post_frame_dir = glob.glob(f'{self.config.POST_VID_FRAME_DIR}/*')

    def _color_correct_to_input(self, content, generated):
        # image manipulations for compatibility with opencv
        content = np.array((content * self.MAX_CHANNEL_INTENSITY), dtype=np.float32)
        content = cv2.cvtColor(content, cv2.COLOR_BGR2YCR_CB)
        generated = np.array((generated * self.MAX_CHANNEL_INTENSITY), dtype=np.float32)
        generated = cv2.cvtColor(generated, cv2.COLOR_BGR2YCR_CB)
        generated = self._trim_img(generated)
        # extract channels, merge intensity and color spaces
        color_corrected = np.zeros(generated.shape, dtype=np.float32)
        color_corrected[:, :, 0] = generated[:, :, 0]
        color_corrected[:, :, 1] = content[:, :, 1]
        color_corrected[:, :, 2] = content[:, :, 2]
        return cv2.cvtColor(color_corrected, cv2.COLOR_YCrCb2BGR) / self.MAX_CHANNEL_INTENSITY


    def generate_stylized_video(self):
        self.post_frame_dir = glob.glob(f'{self.config.POST_VID_FRAME_DIR}/*')
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        video_writer = cv2.VideoWriter(self.config.POST_VID_FRAME_PATH, fourcc, self.config.FPS, (self.frame_width, self.config.FRAME_HEIGHT))

        for count, filename in enumerate(sorted(self.post_frame_dir)):
            if count % 10 == 0:
                print(f"Saving frame: {(count/len(self.post_frame_dir)):.0%}")
            image = cv2.imread(filename)
            video_writer.write(image)

        video_writer.release()
        print(f"Style transfer complete! Output at {self.config.POST_VID_PATH}")
    
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