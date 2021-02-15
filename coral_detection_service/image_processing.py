from PIL import Image, ImageDraw
import numpy as np
import collections
import pathlib

from pycoral.adapters import common
from pycoral.adapters import detect
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter

Object = collections.namedtuple('Object', ['label', 'score', 'bbox'])
DEBUG_SHOW_IMAGE = False
DRAW_BOXES = False


def load_model_interpreter(model):
    abs_model_path = _relative_path(model)
    try:
        interpreter = make_interpreter(abs_model_path)
        interpreter.allocate_tensors()
    except ValueError as e:
        raise e(f'Failed to load interpreter model from path {abs_model_path}')
    return interpreter

def tiles_location_gen(img_size, tile_size, overlap):
    """Generates location of tiles after splitting the given image according the tile_size and overlap.

    Args:
      img_size (int, int): size of original image as width x height.
      tile_size (int, int): size of the returned tiles as width x height.
      overlap (int): The number of pixels to overlap the tiles.

    Yields:
      A list of points representing the coordinates of the tile in xmin, ymin,
      xmax, ymax.
    """

    tile_width, tile_height = tile_size
    img_width, img_height = img_size
    h_stride = tile_height - overlap
    w_stride = tile_width - overlap
    for h in range(0, img_height, h_stride):
        for w in range(0, img_width, w_stride):
            xmin = w
            ymin = h
            xmax = min(img_width, w + tile_width)
            ymax = min(img_height, h + tile_height)
            yield [xmin, ymin, xmax, ymax]


def non_max_suppression(objects, threshold):
    """Returns a list of indexes of objects passing the NMS.

    Args:
      objects: result candidates.
      threshold: the threshold of overlapping IoU to merge the boxes.

    Returns:
      A list of indexes containings the objects that pass the NMS.
    """
    if len(objects) == 1:
        return [0]

    boxes = np.array([o.bbox for o in objects])
    xmins = boxes[:, 0]
    ymins = boxes[:, 1]
    xmaxs = boxes[:, 2]
    ymaxs = boxes[:, 3]

    areas = (xmaxs - xmins) * (ymaxs - ymins)
    scores = [o.score for o in objects]
    idxs = np.argsort(scores)

    selected_idxs = []
    while idxs.size != 0:

        selected_idx = idxs[-1]
        selected_idxs.append(selected_idx)

        overlapped_xmins = np.maximum(xmins[selected_idx], xmins[idxs[:-1]])
        overlapped_ymins = np.maximum(ymins[selected_idx], ymins[idxs[:-1]])
        overlapped_xmaxs = np.minimum(xmaxs[selected_idx], xmaxs[idxs[:-1]])
        overlapped_ymaxs = np.minimum(ymaxs[selected_idx], ymaxs[idxs[:-1]])

        w = np.maximum(0, overlapped_xmaxs - overlapped_xmins)
        h = np.maximum(0, overlapped_ymaxs - overlapped_ymins)

        intersections = w * h
        unions = areas[idxs[:-1]] + areas[selected_idx] - intersections
        ious = intersections / unions

        idxs = np.delete(
            idxs, np.concatenate(([len(idxs) - 1], np.where(ious > threshold)[0])))

    return selected_idxs


def draw_object(draw, obj):
    """Draws detection candidate on the image.

    Args:
      draw: the PIL.ImageDraw object that draw on the image.
      obj: The detection candidate.
    """
    draw.rectangle(obj.bbox, outline='red')
    draw.text((obj.bbox[0], obj.bbox[3]), obj.label, fill='#0000')
    draw.text((obj.bbox[0], obj.bbox[3] + 10), str(obj.score), fill='#0000')


def reposition_bounding_box(bbox, tile_location):
    """Relocates bbox to the relative location to the original image.

    Args:
      bbox (int, int, int, int): bounding box relative to tile_location as xmin,
        ymin, xmax, ymax.
      tile_location (int, int, int, int): tile_location in the original image as
        xmin, ymin, xmax, ymax.

    Returns:
      A list of points representing the location of the bounding box relative to
      the original image as xmin, ymin, xmax, ymax.
    """
    bbox[0] = bbox[0] + tile_location[0]
    bbox[1] = bbox[1] + tile_location[1]
    bbox[2] = bbox[2] + tile_location[0]
    bbox[3] = bbox[3] + tile_location[1]
    return bbox

def coral_object_detection(img, model=None, labels=None, interpreter=None, threshold=0.4, sizes=None, overlap=15, iou=0.2, **kwargs):
    if model is None:
        raise ValueError('Model path not given.')
    if labels is None:
        raise ValueError('Label path not given.')

    if interpreter is None:
        interpreter = load_model_interpreter(model)

    labels = read_label_file(_relative_path(labels)) if labels else {}

    objects_by_label = dict()
    img_size = img.size
    if sizes is None:
        sizes = [img_size]
        resize_mode = Image.ANTIALIAS
    else:
        resize_mode = Image.NEAREST
        try:
            sizes = [float(tile_size) for tile_size in sizes.split(',')]
            sizes = [[int(sz * tile_scale) for sz in img_size] for tile_scale in sizes]
        except:
            sizes = [map(int, tile_size.split('x')) for tile_size in sizes.split(',')]

    for tile_size in sizes:
        for tile_location in tiles_location_gen(img_size, tile_size, overlap):
            tile = img.crop(tile_location)
            _, scale = common.set_resized_input(
                interpreter, tile.size,
                lambda size, img=tile: img.resize(size, resize_mode))
            interpreter.invoke()
            objs = detect.get_objects(interpreter, threshold, scale)

            for obj in objs:
                bbox = [obj.bbox.xmin, obj.bbox.ymin, obj.bbox.xmax, obj.bbox.ymax]
                bbox = reposition_bounding_box(bbox, tile_location)

                label = labels.get(obj.id, '')
                objects_by_label.setdefault(label, []).append(Object(label, obj.score, bbox))

    if iou < 1:
        for label, objects in objects_by_label.items():
            idxs = non_max_suppression(objects, iou)
            objects_by_label[label] = [objects[idx] for idx in idxs]

    return objects_by_label

def filter_labels_of_interest(objects_by_label, labels_of_interest):
    return {label: objects_by_label[label] for label in map(lambda lbl: lbl.lower(), labels_of_interest.split(',')) if label in objects_by_label}

def detection_dict_to_list(objects_by_label):
    detections = []
    for label, objects in objects_by_label.items():
        detections.extend([detection_obj_to_dict(label=label, confidence=obj.score, bbox=obj.bbox) for obj in objects])
    return detections

def detection_obj_to_dict(label='', confidence=0.0, bbox=[0, 0, 0, 0]):
    return {'label': label,
            'confidence': confidence,
            'x_min': bbox[0],
            'y_min': bbox[1],
            'x_max': bbox[2],
            'y_max': bbox[3]
            }

def draw_detection_boxes(image, objects_by_label):
    draw = ImageDraw.Draw(image)
    for label, objects in objects_by_label.items():
        for curr_object in objects:
            draw_object(draw, curr_object)
    if DEBUG_SHOW_IMAGE:
        image.show()
    return image

def _relative_path(pth):
    return pathlib.Path(__file__).parent.joinpath(pth).absolute().as_posix()

def compute(image, detection_params={}, detect='', **kwargs):
    try:
        # Process PIL Image here
        image = image if isinstance(image, Image.Image) else Image.open(input).convert('RGB')
        objects_by_label = coral_object_detection(image, model=kwargs['model'], labels=kwargs['labels'], interpreter=kwargs.get('interpreter', None), **detection_params)
        if detect:
            objects_by_label = filter_labels_of_interest(objects_by_label, detect)
        predictions = detection_dict_to_list(objects_by_label)
        if DRAW_BOXES:
            draw_detection_boxes(image, objects_by_label)
    except:
        predictions = None
    return predictions

