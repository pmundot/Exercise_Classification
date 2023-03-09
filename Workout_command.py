import argparse
import mimetypes
from workout_classification import photo_classification, video_classification



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Workout Classification')
    parser.add_argument('Command', metavar='<command>', choices=['print'], type=str, help='Print only')
    parser.add_argument('-i','--input_file',dest='do_input',metavar='<input>',default=None,help="Enter file path to photo or video. If using webcam enter 0.")
    parser.add_argument('-o','--output_file',dest='do_output', metavar='<output>',default=None,help="Will output a file with given name")
    parser.add_argument('-l','--layer',dest='do_layer', metavar='<layer>',default='default',choices=['default','skeleton','angels','classification'],help='How the file shoule be saved')
    args = parser.parse_args()

    if mimetypes.guess_type(args.do_input)[0].startswith('image'):
        ex = photo_classification(args.do_input)
        ex.class_image(args.do_layer)

    if mimetypes.guess_type(args.do_input)[0].startswith('video'):
        ex = video_classification(args.do_input)
        ex.class_video(args.do_layer)
    else:
        print("Not a valid file type. Please try again")