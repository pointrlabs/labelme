import base64
import contextlib
import io
import json
import os.path as osp
from enum import Enum

import PIL.Image

from qtpy import QtGui

from labelme import __version__
from labelme.logger import logger
from labelme import PY2
from labelme import QT4
from labelme import utils
from .utils.yolo_io import YOLOWriter, YoloReader


PIL.Image.MAX_IMAGE_PIXELS = None


@contextlib.contextmanager
def open(name, mode):
    assert mode in ["r", "w"]
    if PY2:
        mode += "b"
        encoding = None
    else:
        encoding = "utf-8"
    yield io.open(name, mode, encoding=encoding)
    return


class LabelFileFormat(Enum):
    JSON = 'JSON'
    YOLO = 'YOLO'

class LabelFileError(Exception):
    pass


class LabelFile(object):

    suffix = ".txt"

    def __init__(self, filename=None, default_predef_classes_file=None):
        self.shapes = []
        self.imagePath = None
        self.imageData = None
        self.default_predef_classes_file = default_predef_classes_file
        self.predef_classes_file = None
        if filename is not None:
            if self.suffix == ".txt":
                self.load_yolo_file(filename)
            elif self.suffix == ".json":
                self.load(filename)
        self.filename = filename

    @staticmethod
    def load_image_file(filename):
        try:
            image_pil = PIL.Image.open(filename)
        except IOError:
            logger.error("Failed opening image file: {}".format(filename))
            return

        # apply orientation to image according to exif
        image_pil = utils.apply_exif_orientation(image_pil)

        with io.BytesIO() as f:
            ext = osp.splitext(filename)[1].lower()
            if PY2 and QT4:
                format = "PNG"
            elif ext in [".jpg", ".jpeg"]:
                format = "JPEG"
            else:
                format = "PNG"
            image_pil.save(f, format=format)
            f.seek(0)
            return f.read()

    def load(self, filename):
        keys = [
            "version",
            "imageData",
            "imagePath",
            "shapes",  # polygonal annotations
            "flags",  # image level flags
            "imageHeight",
            "imageWidth",
        ]
        shape_keys = [
            "label",
            "points",
            "group_id",
            "shape_type",
            "flags",
        ]
        try:
            with open(filename, "r") as f:
                data = json.load(f)
            version = data.get("version")
            if version is None:
                logger.warn(
                    "Loading JSON file ({}) of unknown version".format(
                        filename
                    )
                )
            elif version.split(".")[0] != __version__.split(".")[0]:
                logger.warn(
                    "This JSON file ({}) may be incompatible with "
                    "current labelme. version in file: {}, "
                    "current version: {}".format(
                        filename, version, __version__
                    )
                )

            if data["imageData"] is not None:
                imageData = base64.b64decode(data["imageData"])
                if PY2 and QT4:
                    imageData = utils.img_data_to_png_data(imageData)
            else:
                # relative path from label file to relative path from cwd
                imagePath = osp.join(osp.dirname(filename), data["imagePath"])
                imageData = self.load_image_file(imagePath)
            flags = data.get("flags") or {}
            imagePath = data["imagePath"]
            self._check_image_height_and_width(
                base64.b64encode(imageData).decode("utf-8"),
                data.get("imageHeight"),
                data.get("imageWidth"),
            )
            shapes = [
                dict(
                    label=s["label"],
                    points=s["points"],
                    shape_type=s.get("shape_type", "polygon"),
                    flags=s.get("flags", {}),
                    group_id=s.get("group_id"),
                    other_data={
                        k: v for k, v in s.items() if k not in shape_keys
                    },
                )
                for s in data["shapes"]
            ]
        except Exception as e:
            raise LabelFileError(e)

        predef_classes_file = osp.join(osp.dirname(osp.realpath(filename)), "classes.txt")
        if not osp.exists(predef_classes_file):
            predef_classes_file = self.default_predef_classes_file

        otherData = {}
        for key, value in data.items():
            if key not in keys:
                otherData[key] = value

        # Only replace data after everything is loaded.
        self.flags = flags
        self.shapes = shapes
        self.imagePath = imagePath
        self.imageData = imageData
        self.filename = filename
        self.otherData = otherData
        self.predef_classes_file = predef_classes_file

    def load_yolo_file(self, filename):
        if osp.isfile(filename) is False:
            return

        imagePath = osp.splitext(filename)[0] + ".png"
        imageData = self.load_image_file(imagePath)
        image = QtGui.QImage.fromData(imageData)
        yolo_reader = YoloReader(filename, image, self.default_predef_classes_file)
        shapes = yolo_reader.get_shapes()

        # Only replace data after everything is loaded.
        self.flags = {}
        self.shapes = shapes
        self.imagePath = imagePath
        self.imageData = imageData
        self.filename = filename
        self.otherData = {}
        self.predef_classes_file = yolo_reader.class_list_path

    @staticmethod
    def _check_image_height_and_width(imageData, imageHeight, imageWidth):
        img_arr = utils.img_b64_to_arr(imageData)
        if imageHeight is not None and img_arr.shape[0] != imageHeight:
            logger.error(
                "imageHeight does not match with imageData or imagePath, "
                "so getting imageHeight from actual image."
            )
            imageHeight = img_arr.shape[0]
        if imageWidth is not None and img_arr.shape[1] != imageWidth:
            logger.error(
                "imageWidth does not match with imageData or imagePath, "
                "so getting imageWidth from actual image."
            )
            imageWidth = img_arr.shape[1]
        return imageHeight, imageWidth

    def save(
        self,
        filename,
        shapes,
        imagePath,
        imageHeight,
        imageWidth,
        imageData=None,
        otherData=None,
        flags=None,
    ):
        fileDir = osp.dirname(filename)
        filePathBase = osp.splitext(osp.basename(filename))[0]
        filePath = osp.join(fileDir, filePathBase + ".json")
        if imageData is not None:
            imageData = base64.b64encode(imageData).decode("utf-8")
            imageHeight, imageWidth = self._check_image_height_and_width(
                imageData, imageHeight, imageWidth
            )
        if otherData is None:
            otherData = {}
        if flags is None:
            flags = {}
        data = dict(
            version=__version__,
            flags=flags,
            shapes=shapes,
            imagePath=imagePath,
            imageData=imageData,
            imageHeight=imageHeight,
            imageWidth=imageWidth,
        )
        for key, value in otherData.items():
            assert key not in data
            data[key] = value
        try:
            with open(filePath, "w") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            self.filename = filename
        except Exception as e:
            raise LabelFileError(e)

    def save_yolo_format(self, filename, shapes, image_path, image_data, class_list):
        fileDir = osp.dirname(filename)
        filePathBase = osp.splitext(osp.basename(filename))[0]
        filePath = osp.join(fileDir, filePathBase + ".txt")
        img_folder_path = osp.dirname(image_path)
        img_folder_name = osp.split(img_folder_path)[-1]
        img_file_name = osp.basename(image_path)
        if isinstance(image_data, QtGui.QImage):
            image = image_data
        else:
            image_data = self.load_image_file(image_path)
            image = QtGui.QImage.fromData(image_data)
        image_shape = [image.height(), image.width(),
                       1 if image.isGrayscale() else 3]
        writer = YOLOWriter(img_folder_name, img_file_name,
                            image_shape, local_img_path=image_path)

        for shape in shapes:
            if shape['shape_type'] != 'rectangle':
                continue

            points = shape['points']
            label = shape['label']
            # Add Chris
            bnd_box = LabelFile.convert_points_to_bnd_box(points)
            writer.add_bnd_box(bnd_box[0], bnd_box[1], bnd_box[2], bnd_box[3], label)

        writer.save(target_file=filePath, class_list=class_list)
        return


    @staticmethod
    def is_label_file(filename):
        return osp.splitext(filename)[1].lower() == LabelFile.suffix

    @staticmethod
    def convert_points_to_bnd_box(points):
        x_min = float('inf')
        y_min = float('inf')
        x_max = float('-inf')
        y_max = float('-inf')
        for p in points:
            x = p[0]
            y = p[1]
            x_min = min(x, x_min)
            y_min = min(y, y_min)
            x_max = max(x, x_max)
            y_max = max(y, y_max)

        # Martin Kersner, 2015/11/12
        # 0-valued coordinates of BB caused an error while
        # training faster-rcnn object detector.
        if x_min < 1:
            x_min = 1

        if y_min < 1:
            y_min = 1

        return int(x_min), int(y_min), int(x_max), int(y_max)

