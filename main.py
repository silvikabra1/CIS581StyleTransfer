from config import Config

class StyleTransfer:
    def __init__(self, config=Config):
        self.config = config

    def run(self):
        print("Fetching input frames")
        self.fetch_input_frames()
        print("Fetching style reference info")
        self.fetch_style_refs()
        print("Fetching output frames")
        self.fetch_output_frames()
        print("Saving video")
        self.generate_stylized_video()

## Run as a script
if __name__ == "__main__":
    StyleTransfer().run()