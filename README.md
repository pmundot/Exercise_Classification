# Workout Classifier

This project is designed to take in a photo or video of a workout and classify if it is a squat, push up, or jumping jack. The program can be run through the command line or by using the provided Jupyter notebook in the "background" folder. The program is built using the libraries sklearn, mediapipe, opencv, and mimetypes.

## Installation

To run this program, you will need to have Python 3 installed on your computer. Additionally, you will need to install the following libraries:

- scikit-learn
- mediapipe
- opencv
- mimetypes
- pickle
- math

You can install these libraries by running the following command in your terminal:

```shell
pip install scikit-learn mediapipe opencv-python mimetypes pickle math
```

## Usage

To use the program, navigate to the directory where the repository is stored in your terminal.

There are 3 inputs for the program:

1.  `-i input_file`: the path to the video or photo to be classified.
2.  `-o output_file`: the name of the image you would like to classify and save. If left blank, the image will not be saved and will only be shown. You must enter file type as well.
3.  `-l layer`: which of 4 layers to choose from. The 4 layers are:
   - `Default`: which is only the classification name.
   - `Angles`: which shows the angles on images.
   - `Skeleton`: which shows the Mediapipe format.
   - `Classification`: which shows all of the above.

To run the program through the command line, type:

```shell
python workout_command.py print -i<input_file> -o<output_file> -l<layer>
```

example provided with file test.jpg
```shell
python workout_command.py print -i test.jpg -o classified.jpg 
```

Replace `<input_file>` with the path to the image or video you would like to classify. Replace `<output_file>` with the name of the image you would like to classify and save along with file type. If you do not want to save the image, leave `<output_file>` blank. Replace `<layer>` with one of the 4 layers listed above if left blank then default will be used.

To use the Jupyter notebook, open the "background" folder and launch the "workout_classifier.ipynb" file. Follow the instructions provided in the notebook to classify an image or video.

## Ending Video early

If running a video and you wish to end the video early, press q on your keyboard.

## Data

The data used to train this program is self-generated and stored in the "background" folder.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
