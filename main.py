from config import Config
import os
import glob
import cv2
import numpy as np
import tensorflow_hub as hub
import tensorflow as tf
from getpass import getpass
import openai
import requests

import styletransfer
import importlib
importlib.reload(styletransfer)


class StyleTransfer:

    MAX_CHANNEL_INTENSITY = 255.0

    def __init__(self, uploaded_file=None, prompt=None, config=Config):
        # set up config with uploaded video
        if uploaded_file is None:
            raise ValueError(f"Error: No video provided")
        if prompt is None:
            raise ValueError(f"Error: No prompt provided")
        self.config = config
        self.config.PRE_VID_NAME = list(uploaded_file.keys())[0]
        self.prompt = prompt
        #self.hub_module = hub.load(self.config.TENSORFLOW_HUB_HANDLE)
        self.hub_module = styletransfer.Styler()
        self.pre_frame_dir = glob.glob(f'{self.config.PRE_VID_FRAME_DIR}/*')
        self.post_frame_dir = glob.glob(f'{self.config.POST_VID_FRAME_DIR}/*')

        self.style_dir = glob.glob(f'{self.config.STYLE_DIR}/*')
        delete_these_files = self.style_dir
        if self.config.CLEAR_INPUT_FRAME_CACHE:
            delete_these_files += self.post_frame_dir
            delete_these_files += self.pre_frame_dir

        for file in delete_these_files:
            os.remove(file)

         # Update contents of directory after deletion
        self.pre_frame_dir = glob.glob(f'{self.config.PRE_VID_FRAME_DIR}/*')
        self.post_frame_dir = glob.glob(f'{self.config.POST_VID_FRAME_DIR}/*')

        if len(self.pre_frame_dir) > 0:
            self.frame_width = cv2.imread(self.pre_frame_dir[0]).shape[1]

    def fetch_pre_frames(self):
        if len(self.pre_frame_dir) > 0:
            print("Using cached frames")
            return

        video = cv2.VideoCapture(self.config.PRE_VID_PATH)
        frame_interval = np.floor((1.0 / self.config.FPS) * 1000)
        ret, frame = video.read()

        if frame is None:
            raise ValueError(f"Error: No video provided")

        # Adjust scale based on specified frame height
        scale = self.config.FRAME_HEIGHT / frame.shape[0]
        self.frame_width = int(frame.shape[1] * scale)

        frame = cv2.resize(
            frame, (self.frame_width, self.config.FRAME_HEIGHT)).astype(np.uint8)
        cv2.imwrite(self.config.PRE_VID_FRAME_PATH.format(0), frame)

        # Sample original video at specified frame rate
        offset = 1
        while ret:
            timestamp = offset * frame_interval
            video.set(cv2.CAP_PROP_POS_MSEC, timestamp)
            ret, frame = video.read()
            if not ret:
                break
            frame = cv2.resize(
                frame, (self.frame_width, self.config.FRAME_HEIGHT)).astype(np.uint8)
            cv2.imwrite(self.config.PRE_VID_FRAME_PATH.format(offset), frame)
            offset += 1

        self.pre_frame_dir = glob.glob(f'{self.config.PRE_VID_FRAME_DIR}/*')

    def generate_styles(self):
        # Parse prompt and get image URLs
        style_inputs = self.prompt.split(", ")
        style_inputs = [None if input ==
                        "None" else input for input in style_inputs]
        self.style_inputs = style_inputs
        for idx, input in enumerate(style_inputs):
            if input is None:
                continue
            else:
                response = openai.Image.create(
                    prompt=input,
                    n=1,
                    size="1024x1024"
                )
                image_url = response['data'][0]['url']
                response = requests.get(image_url).content
                file_name = self.config.STYLE_IMG_PATH.format(idx)
                with open(file_name, 'wb') as handler:
                    handler.write(response)

    # input to Jimmy's part: "None, pointilism, None, cubism" - [None, filename, None, filename]
    def fetch_style_refs(self):
        num_frames = len(self.pre_frame_dir)
        style_ref_imgs = list()
        style_ref_img_resized = False
        style_ref_img_files = sorted(self.style_dir)
        self.num_ref_imgs = len(self.style_inputs)
        self.num_frames_per_style = num_frames if self.num_ref_imgs == 1 else np.ceil(
            num_frames / (self.num_ref_imgs - 1))

        # Make all style ref imgs same size as first style
        style_ref_img_1_height = None
        style_ref_img_1_width = None

        curr_style_img_index = 0
        for i in range(self.num_ref_imgs):
            # Check index of style inputs to see if it is None
            if self.style_inputs[i] is None:
                style_ref_imgs.append(None)
                continue
            style_ref_img = cv2.imread(
                style_ref_img_files[curr_style_img_index])
            style_ref_img = cv2.cvtColor(style_ref_img, cv2.COLOR_BGR2RGB)
            if style_ref_img_1_height is None or style_ref_img_1_width is None:
                style_ref_img_1_height, style_ref_img_1_width, channels = style_ref_img.shape
            else:
                style_ref_img_height, style_ref_img_width, channels = style_ref_img.shape
                # Change these style imgs to match first style img
                if style_ref_img_1_height != style_ref_img_height or style_ref_img_1_width != style_ref_img_width:
                    style_ref_img = cv2.resize(
                        style_ref_img, (style_ref_img_1_width, style_ref_img_1_height))
                    style_ref_img_resized = True
            style_ref_imgs.append(style_ref_img / self.MAX_CHANNEL_INTENSITY)
            curr_style_img_index += 1

        self.style_ref_imgs = style_ref_imgs
        # Alert user that style images were resized
        if style_ref_img_resized:
            print("Warning: Resizing style images -> may cause distortion")

    def trim_img(self, img):
        return img[:self.config.FRAME_HEIGHT, :self.frame_width]

    def fetch_post_frames(self):
        self.pre_frame_dir = glob.glob(f'{self.config.PRE_VID_FRAME_DIR}/*')
        ghost_frame = None
        for frame_idx, filename in enumerate(sorted(self.pre_frame_dir)):
            if frame_idx % 10 == 0:
                print(
                    f"Output frame: {(frame_idx/len(self.pre_frame_dir)):.0%}")
            current_frame = cv2.imread(filename)
            current_frame = cv2.cvtColor(
                current_frame, cv2.COLOR_BGR2RGB) / self.MAX_CHANNEL_INTENSITY
            curr_style_img_idx = int(frame_idx / self.num_frames_per_style)
            blend_ratio = 1 - \
                ((frame_idx % self.num_frames_per_style) / self.num_frames_per_style)
            inv_blend_ratio = 1 - blend_ratio

            prev_style = self.style_ref_imgs[curr_style_img_idx]
            next_style = self.style_ref_imgs[curr_style_img_idx + 1]

            # If both are content images, don't need to apply style transfer - TEST out
            if prev_style is None and next_style is None:
                temp_ghost_frame = cv2.cvtColor(
                    ghost_frame, cv2.COLOR_RGB2BGR) * self.MAX_CHANNEL_INTENSITY
                cv2.imwrite(self.config.POST_VID_FRAME_PATH.format(
                    frame_idx), temp_ghost_frame)
                continue

            if frame_idx > 0:
                current_frame = ((1 - self.config.GHOST_FRAME_TRANSPARENCY) * current_frame) + (
                    self.config.GHOST_FRAME_TRANSPARENCY * ghost_frame)
            current_frame = tf.cast(
                tf.convert_to_tensor(current_frame), tf.float32)

            # if previous img is not a style img, then next must be a style img, which will be what we blend
            if prev_style is None:
                img_to_blend = next_style
            # if next img is not a style img, then prev must be a style img, which will be what we blend
            elif next_style is None:
                img_to_blend = prev_style
            else:
                blended_prev_style = blend_ratio * prev_style
                blended_next_style = inv_blend_ratio * next_style
                img_to_blend = blended_prev_style + blended_next_style

            # img_to_blend = tf.cast(tf.convert_to_tensor(img_to_blend), tf.float32)
            # expanded_blended_img = tf.constant(tf.expand_dims(img_to_blend, axis=0))
            # expanded_current_frame = tf.constant(tf.expand_dims(current_frame, axis=0))

            # Apply style transfer
            stylized_frame = self.hub_module.apply_style(
                current_frame, img_to_blend)
            stylized_frame = tf.squeeze(stylized_frame.detach().numpy())

            stylized_frame = tf.transpose(stylized_frame, perm=[1, 2, 0])
            a, b, _ = stylized_frame.get_shape()
            modified_current_frame = tf.identity(current_frame)
            modified_current_frame = tf.image.resize(
                modified_current_frame, size=(a, b))

            # Re-blend if one of the images is a content image
            if prev_style is None:
                reblended_prev_style = blend_ratio * modified_current_frame
                reblended_next_style = inv_blend_ratio * stylized_frame
            if next_style is None:
                reblended_prev_style = blend_ratio * stylized_frame
                reblended_next_style = inv_blend_ratio * modified_current_frame
            if prev_style is None or next_style is None:
                stylized_frame = self.trim_img(
                    reblended_prev_style) + self.trim_img(reblended_next_style)

            if self.config.KEEP_COLORS:
                stylized_frame = self.color_correct_to_input(
                    modified_current_frame, stylized_frame)

            a, b, _ = current_frame.get_shape()
            modified_stylized_frame = tf.identity(stylized_frame)
            modified_stylized_frame = tf.image.resize(
                modified_stylized_frame, size=(a, b))
            ghost_frame = np.asarray(self.trim_img(modified_stylized_frame))

            temp_ghost_frame = cv2.cvtColor(
                ghost_frame, cv2.COLOR_RGB2BGR) * self.MAX_CHANNEL_INTENSITY
            cv2.imwrite(self.config.POST_VID_FRAME_PATH.format(
                frame_idx), temp_ghost_frame)
        self.post_frame_dir = glob.glob(f'{self.config.POST_VID_FRAME_DIR}/*')

    def color_correct_to_input(self, input, precorrected_output):
        # Change image so that it is compatible with OpenCV
        input = np.array((input * self.MAX_CHANNEL_INTENSITY),
                         dtype=np.float32)
        input = cv2.cvtColor(input, cv2.COLOR_BGR2YCR_CB)
        pre_corrected_output = np.array(
            (pre_corrected_output * self.MAX_CHANNEL_INTENSITY), dtype=np.float32)
        pre_corrected_output = cv2.cvtColor(
            pre_corrected_output, cv2.COLOR_BGR2YCR_CB)
        pre_corrected_output = self.trim_img(pre_corrected_output)
        # Extract the channels, merge the intensities and color spaces
        color_corrected_output = np.zeros(
            pre_corrected_output.shape, dtype=np.float32)
        color_corrected_output[:, :, 0] = pre_corrected_output[:, :, 0]
        color_corrected_output[:, :, 1] = pre_corrected_output[:, :, 1]
        color_corrected_output[:, :, 2] = pre_corrected_output[:, :, 2]
        return cv2.cvtColor(color_corrected_output, cv2.COLOR_YCrCb2BGR) / self.MAX_CHANNEL_INTENSITY

    def generate_stylized_video(self):
        self.post_frame_dir = glob.glob(f'{self.config.POST_VID_FRAME_DIR}/*')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # print("frame width", self.frame_width)
        video_writer = cv2.VideoWriter(
            self.config.POST_VID_PATH, fourcc, self.config.FPS, (self.frame_width, self.config.FRAME_HEIGHT))

        # Write frames to video
        for count, filename in enumerate(sorted(self.post_frame_dir)):
            if count % 10 == 0:
                print(f"Saving frame: {(count/len(self.post_frame_dir)):.0%}")
            image = cv2.imread(filename)
            video_writer.write(image)

        video_writer.release()
        print(
            f"Style transfer complete! Output at {self.config.POST_VID_PATH}")

    def run(self):
        print("Generating style references")
        self.generate_styles()
        print("Fetching input frames")
        self.fetch_pre_frames()
        print("Fetching style reference info")
        self.fetch_style_refs()
        print("Fetching output frames")
        self.fetch_post_frames()
        print("Saving video")
        self.generate_stylized_video()


# Run as a script
if __name__ == "__main__":
    StyleTransfer().run()
