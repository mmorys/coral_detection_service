import argparse
from coral_detection_service.server import Server

def coral_serve():
    parser = argparse.ArgumentParser(description="Run a simple HTTP server")
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default='./models/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite',
        help="Path to the TF model for detection",
    )
    parser.add_argument(
        "-l",
        "--labels",
        type=str,
        default='./models/ssd_mobilenet_v2_coco_labels.txt',
        help="Path to the TF model labels text file",
    )
    parser.add_argument(
        "-a",
        "--address",
        type=str,
        default="localhost",
        help="Specify the IP address on which the server listens",
    )
    parser.add_argument(
        "-p",
        "--port",
        type=int,
        default=8000,
        help="Specify the port on which the server listens",
    )
    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=0.4,
        help="Detection confidence threshold between 0 and 1 above which to declare detections",
    )
    parser.add_argument(
        "-i",
        "--iou",
        type=float,
        default=0.1,
        help="IoU threshold between 0 and 1 for non max suppression. Set to 1 to disable non-max suppression",
    )
    parser.add_argument(
        "-o",
        "--overlap",
        type=int,
        default=15,
        help="If using tiled detection, nmumber of pixels by which to overlap tiles",
    )
    parser.add_argument(
        "-s",
        "--sizes",
        type=str,
        default='1',
        help="""String of tile sizes to use for detection, concatenated with commas.
        Can be specified as a list of fractions relative to the image size, e.g. '1,0.5,0.25'.
        Can also be specified as absolute values of pixels, e.g. '600x400,320x240'.
        Setting to '1' will perform detection on a single tile the size of the original image.
        """
    )
    parser.add_argument(
        "-d",
        "--detect",
        type=str,
        default='',
        help="Comma separated string of labels to detect in output. If empty, all labels will be detected. e.g. 'person,dog'",
    )

    args = parser.parse_args()
    detection_params = {'threshold': args.threshold,
                        'iou': args.iou,
                        'overlap': args.overlap,
                        'sizes': args.sizes}
    arg_params = {'detection_params': detection_params,
                  'detect': args.detect,
                  'model': args.model,
                  'labels': args.labels}

    for arg in vars(args):
        print(f'{arg}: {getattr(args, arg)}')

    Server().run(addr=args.address, port=args.port, arg_params=arg_params)

if __name__ == '__main__':
    coral_serve()