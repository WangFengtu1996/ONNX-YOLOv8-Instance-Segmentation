import math
import time
import cv2
import numpy as np
import onnxruntime
import pprint

from yoloseg.utils import xywh2xyxy, nms, draw_detections, sigmoid, save_by_plt

pprint = pprint.PrettyPrinter(width=10000).pprint
np.set_printoptions(threshold=np.inf)
class YOLOSeg:

    def __init__(self, path, conf_thres=0.7, iou_thres=0.5, num_masks=32, resize=False):
        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres
        self.num_masks = num_masks
        self.resize = resize

        # Initialize model
        self.initialize_model(path)

    def __call__(self, image):
        return self.segment_objects(image)

    def initialize_model(self, path):
        # self.session = onnxruntime.InferenceSession(path,
        #                                             providers=['TensorrtExecutionProvider',
        #                                                        'CUDAExecutionProvider',
        #                                                        'CPUExecutionProvider'])
        self.session = onnxruntime.InferenceSession(path,providers=['CUDAExecutionProvider',
                                        'CPUExecutionProvider'])
        # Get model info
        self.get_input_details()
        self.get_output_details()

    def segment_objects(self, image):
        #
        if self.resize:
            input_tensor = self.prepare_input_resize(image)
        else:
            input_tensor = self.prepare_input(image)

        # Perform inference on the image
        outputs = self.inference(input_tensor)
        print("*" * 70)
        shapes = [np.squeeze(item).shape for item in outputs]
        pprint(shapes)
        # pprint(np.array(outputs).shape())
        print("*" * 70)
        self.boxes, self.scores, self.class_ids, mask_pred = self.process_box_output(outputs[0])

        if self.resize:
        #
            self.mask_maps = self.process_mask_output_resize(mask_pred, outputs[1])
        else:
            self.mask_maps = self.process_mask_output(mask_pred, outputs[1])

        return self.boxes, self.scores, self.class_ids, self.mask_maps
        # return self.boxes, self.scores, self.class_ids, None

    def prepare_input(self, image):
        self.img_height, self.img_width = image.shape[:2]

        input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if(self.img_height > self.img_width):
            self.crop_height = self.input_height
            self.crop_width = self.input_width * (self.img_width * 1.0 /  self.img_height)
        else:
            self.crop_width = self.input_width
            self.crop_height = self.input_height * ( self.img_height  * 1.0 / self.img_width)

        print(self.crop_height, self.crop_width)
        self.crop_width, self.crop_height = int(self.crop_width), int(self.crop_height)
        # Resize input image
        input_img = cv2.resize(input_img, (self.crop_width, self.crop_height))
        convert_img = cv2.cvtColor(input_img, cv2.COLOR_RGB2BGR)
        # cv2.imwrite("resize.jpg", convert_img)
        print(input_img.shape)
        # Scale input pixel values to 0 to 1
        input_img = input_img / 255.0
        input_padding_img = np.ones(((self.input_width, self.input_height, input_img.shape[-1])), dtype=np.float32)
        input_padding_img[:input_img.shape[0],:,:] = input_img
        # input_padding_img_show = cv2.cvtColor(input_padding_img, cv2.COLOR_RGB2BGR)
        # cv2.imshow("input_padding_img_show", input_padding_img_show)
        input_padding_img = input_padding_img.transpose(2, 0, 1)
        input_tensor = input_padding_img[np.newaxis, :, :, :].astype(np.float32)

        return input_tensor

    def prepare_input_resize(self, image):
        self.img_height, self.img_width = image.shape[:2]

        input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize input image
        input_img = cv2.resize(input_img, (self.input_width, self.input_height))

        # Scale input pixel values to 0 to 1
        input_img = input_img / 255.0

        input_img = input_img.transpose(2, 0, 1)
        input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32)
        self.crop_width, self.crop_height = int(self.input_width), int(self.input_height)
        return input_tensor

    def inference(self, input_tensor):
        start = time.perf_counter()
        outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor})

        # print(f"Inference time: {(time.perf_counter() - start)*1000:.2f} ms")
        return outputs

    def process_box_output(self, box_output):

        predictions = np.squeeze(box_output).T
        num_classes = box_output.shape[1] - self.num_masks - 4
        print('zero predictions shape : ', predictions.shape)
        # Filter out object confidence scores below threshold
        scores = np.max(predictions[:, 4:4+num_classes], axis=1)
        print("scores shape ",scores.shape)
        print("scores > self.conf_threshold shape", (scores > self.conf_threshold).shape)
        predictions = predictions[scores > self.conf_threshold, :]
        print("predictions : ", predictions.shape)
        scores = scores[scores > self.conf_threshold]
        print("second scores shape : ", scores.shape)
        if len(scores) == 0:
            return [], [], [], np.array([])

        box_predictions = predictions[..., :num_classes+4]
        mask_predictions = predictions[..., num_classes+4:]

        # Get the class with the highest confidence
        class_ids = np.argmax(box_predictions[:, 4:], axis=1)

        # Get bounding boxes for each object
        boxes = self.extract_boxes(box_predictions)

        # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
        indices = nms(boxes, scores, self.iou_threshold)

        return boxes[indices], scores[indices], class_ids[indices], mask_predictions[indices]

    def process_mask_output(self, mask_predictions, mask_output):

        if mask_predictions.shape[0] == 0:
            return []

        mask_output = np.squeeze(mask_output)

        # Calculate the mask maps for each box
        num_mask, mask_height, mask_width = mask_output.shape  # CHW
        print("num_mask : {} mask_height : {} mask_width : {}".format(num_mask, mask_height, mask_width))
        # save_by_plt(mask_predictions.reshape(-1),mask_predictions.reshape(-1), "mask_predictions.png")
        # save_by_plt(mask_output.reshape(-1),mask_output.reshape(-1), "mask_output.png")
        masks = sigmoid(mask_predictions @ mask_output.reshape((num_mask, -1)))
        print("mask_predictions shape : {} mask_output.reshape : {} masks shape : {}".format(mask_predictions.shape, mask_output.reshape((num_mask, -1)).shape, masks.shape) )
        masks = masks.reshape((-1, mask_height, mask_width))
        print("masks vale " ,  (masks >= 0).all() and (masks <= 1).all())

        y = masks.reshape(-1)
        y = y[y>0.5]
        # x = np.zeros(len(y))
        print(y.shape)
        # save_by_plt(y,y, 'mask.png')


        if(self.img_height > self.img_width):
            self.input_pad_height = self.img_height
            self.input_pad_width = self.img_width * ( self.img_height* 1.0 /  self.img_width)
        else:
            self.input_pad_width = self.img_width
            self.input_pad_height = self.img_height * ( self.img_width  * 1.0 / self.img_height)

        print(self.input_pad_height, self.input_pad_width)
        self.input_pad_width, self.input_pad_height = int(self.input_pad_width), int(self.input_pad_height)

        # Downscale the boxes to match the mask size
        scale_boxes = self.rescale_boxes(self.boxes,
                                   (self.input_pad_height, self.input_pad_width),
                                   (mask_height, mask_width))

        # For every box/mask pair, get the mask map
        mask_maps = np.zeros((len(scale_boxes), self.input_pad_height, self.input_pad_width))
        blur_size = (int(self.input_pad_width / mask_width), int(self.input_pad_height / mask_height))
        for i in range(len(scale_boxes)):

            scale_x1 = int(math.floor(scale_boxes[i][0]))
            scale_y1 = int(math.floor(scale_boxes[i][1]))
            scale_x2 = int(math.ceil(scale_boxes[i][2]))
            scale_y2 = int(math.ceil(scale_boxes[i][3]))

            x1 = int(math.floor(self.boxes[i][0]))
            y1 = int(math.floor(self.boxes[i][1]))
            x2 = int(math.ceil(self.boxes[i][2]))
            y2 = int(math.ceil(self.boxes[i][3]))

            scale_crop_mask = masks[i][scale_y1:scale_y2, scale_x1:scale_x2]
            crop_mask = cv2.resize(scale_crop_mask,
                              (x2 - x1, y2 - y1),
                              interpolation=cv2.INTER_CUBIC)

            crop_mask = cv2.blur(crop_mask, blur_size)
            # print("mask vale " ,  (crop_mask > 0).all() and (crop_mask < 1).all())
            # pprint(crop_mask)

            crop_mask = (crop_mask > 0.5).astype(np.uint8)
            # if i == 0:
            #     np.savetxt('object1.txt', crop_mask, delimiter=',')
            # print(str(i))
            # pprint(crop_mask)
            # with open('mask.txt', 'a') as w:
            #     w.write(np.array2string(crop_mask))
            mask_maps[i, y1:y2, x1:x2] = crop_mask

        return mask_maps[:, :self.img_height, :self.img_width]

    def process_mask_output_resize(self, mask_predictions, mask_output):

        if mask_predictions.shape[0] == 0:
            return []

        mask_output = np.squeeze(mask_output)

        # Calculate the mask maps for each box
        num_mask, mask_height, mask_width = mask_output.shape  # CHW
        print("num_mask : {} mask_height : {} mask_width : {}".format(num_mask, mask_height, mask_width))
        save_by_plt(mask_predictions.reshape(-1),mask_predictions.reshape(-1), "mask_predictions.png")
        save_by_plt(mask_output.reshape(-1),mask_output.reshape(-1), "mask_output.png")
        masks = sigmoid(mask_predictions @ mask_output.reshape((num_mask, -1)))
        print("mask_predictions shape : {} mask_output.reshape : {} masks shape : {}".format(mask_predictions.shape, mask_output.reshape((num_mask, -1)).shape, masks.shape) )
        masks = masks.reshape((-1, mask_height, mask_width))
        print("masks vale " ,  (masks >= 0).all() and (masks <= 1).all())

        y = masks.reshape(-1)
        y = y[y>0.5]
        # x = np.zeros(len(y))
        print(y.shape)
        # save_by_plt(y,y, 'mask.png')

        # Downscale the boxes to match the mask size
        scale_boxes = self.rescale_boxes(self.boxes,
                                   (self.img_height, self.img_width),
                                   (mask_height, mask_width))

        # For every box/mask pair, get the mask map
        mask_maps = np.zeros((len(scale_boxes), self.img_height, self.img_width))
        blur_size = (int(self.img_width / mask_width), int(self.img_height / mask_height))
        for i in range(len(scale_boxes)):

            scale_x1 = int(math.floor(scale_boxes[i][0]))
            scale_y1 = int(math.floor(scale_boxes[i][1]))
            scale_x2 = int(math.ceil(scale_boxes[i][2]))
            scale_y2 = int(math.ceil(scale_boxes[i][3]))

            x1 = int(math.floor(self.boxes[i][0]))
            y1 = int(math.floor(self.boxes[i][1]))
            x2 = int(math.ceil(self.boxes[i][2]))
            y2 = int(math.ceil(self.boxes[i][3]))

            scale_crop_mask = masks[i][scale_y1:scale_y2, scale_x1:scale_x2]
            crop_mask = cv2.resize(scale_crop_mask,
                              (x2 - x1, y2 - y1),
                              interpolation=cv2.INTER_CUBIC)

            crop_mask = cv2.blur(crop_mask, blur_size)
            # print("mask vale " ,  (crop_mask > 0).all() and (crop_mask < 1).all())
            # pprint(crop_mask)

            crop_mask = (crop_mask > 0.5).astype(np.uint8)
            # if i == 0:
            #     np.savetxt('object1.txt', crop_mask, delimiter=',')
            # print(str(i))
            # pprint(crop_mask)
            # with open('mask.txt', 'a') as w:
            #     w.write(np.array2string(crop_mask))
            mask_maps[i, y1:y2, x1:x2] = crop_mask

        return mask_maps

    def extract_boxes(self, box_predictions):
        # Extract boxes from predictions
        boxes = box_predictions[:, :4]

        # Scale boxes to original image dimensions

        if self.resize:
            boxes = self.rescale_boxes(boxes,
                                    (self.input_height, self.input_width),
                                    (self.img_height, self.img_width))
        else:
            boxes = self.rescale_boxes(boxes,
                                    (self.input_height, self.input_width),
                                    (self.img_width, self.img_width))
        # Convert boxes to xyxy format
        boxes = xywh2xyxy(boxes)

        # Check the boxes are within the image
        boxes[:, 0] = np.clip(boxes[:, 0], 0, self.img_width)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, self.img_height)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, self.img_width)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, self.img_height)

        return boxes

    def draw_detections(self, image, draw_scores=True, mask_alpha=0.4):
        return draw_detections(image, self.boxes, self.scores,
                               self.class_ids, mask_alpha)

    def draw_masks(self, image, draw_scores=True, mask_alpha=0.5):
        return draw_detections(image, self.boxes, self.scores,
                               self.class_ids, mask_alpha, mask_maps=self.mask_maps)

    def get_input_details(self):
        model_inputs = self.session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]

        self.input_shape = model_inputs[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

    def get_output_details(self):
        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]

    @staticmethod
    def rescale_boxes(boxes, input_shape, image_shape):
        # Rescale boxes to original image dimensions
        input_shape = np.array([input_shape[1], input_shape[0], input_shape[1], input_shape[0]])
        print("input_shape : ", input_shape )
        print("image_shape : ", image_shape)
        # pprint(boxes)
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes *= np.array([image_shape[1], image_shape[0], image_shape[1], image_shape[0]])

        return boxes


if __name__ == '__main__':
    from imread_from_url import imread_from_url

    model_path = "../models/yolov8m-seg.onnx"

    # Initialize YOLOv8 Instance Segmentator
    yoloseg = YOLOSeg(model_path, conf_thres=0.3, iou_thres=0.5)

    img_url = "https://live.staticflickr.com/13/19041780_d6fd803de0_3k.jpg"
    img = imread_from_url(img_url)

    # Detect Objects
    yoloseg(img)

    # Draw detections
    combined_img = yoloseg.draw_masks(img)
    cv2.namedWindow("Output", cv2.WINDOW_NORMAL)
    cv2.imshow("Output", combined_img)
    cv2.waitKey(0)
