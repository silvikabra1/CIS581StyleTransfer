# CIS581StyleTransfer

Instructions to Run:
1. Open this Google Colab <a href="https://drive.google.com/file/d/1qwaKdisfYZYMmifKuIRHvRpZNM-07Er9/view?usp=share_link">link </a> and follow the instructions in the cells. Run through each of them and provide the inputs asked for. If there is an error, restart the runtime and run it again.

Implementation Details:
1. We first use OpenAI to generate style images based on the text that the user input. The order that the user inputs their text is important and determines the order of the styles that the video will have transferred. The user also has the opportunity to include "None" in their inputs if they want the original video to appear for a few frames between the style transferred frames. 
2. We then preprocess the original video and extract the frames using a given frames per second.
3. We then preprocess the generated style images and resize them and convert their color schemes to be consistent with one another.
4. We then generate the output frames of the video. Depending on the user's input from Step 1, we will use the style images generated and perform style transfer with the frames. We blend together the styled images and the original frames proportionally to the number of frames a specific style image has been applied to, fading them in and out to the next style image.
5. We then create the final video using the generated output frames from Step 4.

All of our required packages are listed in requirements.txt and are installed in the first cell of the Google Colab.
