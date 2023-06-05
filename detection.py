#!python3
import argparse
import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '../..')))
import torch
import cv2
import numpy as np
from experiment import Structure, Experiment
from concern.config import Configurable, Config
import math
import PIL.Image


def detection(img_list, c):
    # parser = argparse.ArgumentParser(description='Text Recognition Training')
    # parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    # parser.add_argument('--image_path', type=str, help='image path')
    # parser.add_argument('--result_dir', type=str, default='../Detection/', help='path to save results')
    # parser.add_argument('--data', type=str,
    #                     help='The name of dataloader which will be evaluated on.')
    # parser.add_argument('--image_short_side', type=int, default=736,
    #                     help='The threshold to replace it in the representers')
    # parser.add_argument('--thresh', type=float,
    #                     help='The threshold to replace it in the representers')
    # parser.add_argument('--box_thresh', type=float, default=0.6,
    #                     help='The threshold to replace it in the representers')
    # parser.add_argument('--visualize', action='store_true',
    #                     help='visualize maps in tensorboard')
    # parser.add_argument('--resize', action='store_true',
    #                     help='resize')
    # parser.add_argument('--polygon', action='store_true',
    #                     help='output polygons if true')
    # parser.add_argument('--eager', '--eager_show', action='store_true', dest='eager_show',
    #                     help='Show iamges eagerly')

    # args = {"image_path": img_list, "image_short_side": 1376, "box_thresh": 0.4, "resume": c.MODEL_PATH,
    #         "result_dir": c.DETECTION_RESULTS_PATH, "visualize": True, "polygon": True, "resize": False}
    args = {"image_path": img_list, "image_short_side": 1376, "box_thresh": 0.5, "resume": c.MODEL_PATH,
            "result_dir": c.DETECTION_RESULTS_PATH, "visualize": True, "polygon": True, "resize": False, "save": False}
    # args = vars(args)
    # args = {k: v for k, v in args.items() if v is not None}

    # yaml = os.path.join(c.DBNET_PATH, "experiments/seg_detector/mine_mobilenet_v3_large_thre.yaml")

    conf = Config()
    experiment_args = conf.compile(conf.load(c.YAML_PATH))['Experiment']
    experiment_args.update(cmd=args)
    experiment = Configurable.construct_class_from_config(experiment_args)

    output = Demo(experiment, experiment_args, cmd=args).inference(args['image_path'], args['visualize'],
                                                                   args['save'])
    return output


class Demo:
    def __init__(self, experiment, args, cmd=dict()):
        self.RGB_MEAN = np.array([122.67891434, 116.66876762, 104.00698793])
        self.experiment = experiment
        experiment.load('evaluation', **args)
        self.args = cmd
        model_saver = experiment.train.model_saver
        self.structure = experiment.structure
        self.model_path = self.args['resume']

    def init_torch_tensor(self):
        # Use gpu or not
        torch.set_default_tensor_type('torch.FloatTensor')
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            self.device = torch.device('cpu')

    def init_model(self):
        model = self.structure.builder.build(self.device)
        return model

    def resume(self, model, path):
        if not os.path.exists(path):
            print("Checkpoint not found: " + path)
            return
        print("Resuming from " + path)
        states = torch.load(
            path, map_location=self.device)
        model.load_state_dict(states, strict=False)
        print("Resumed from " + path)

    def resize_image(self, img):
        height, width, _ = img.shape
        if height < width:
            new_height = self.args['image_short_side']
            new_width = int(math.ceil(new_height / height * width / 32) * 32)
        else:
            new_width = self.args['image_short_side']
            new_height = int(math.ceil(new_width / width * height / 32) * 32)
        resized_img = cv2.resize(img, (new_width, new_height))
        return resized_img

    def load_image(self, image_path):
        img = cv2.cvtColor(np.asarray(image_path), cv2.COLOR_RGB2BGR).astype('float32')
        # img = cv2.imread(image_path, cv2.IMREAD_COLOR).astype('float32')
        original_shape = img.shape[:2]
        img = self.resize_image(img)
        img -= self.RGB_MEAN
        img /= 255.
        img = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0)
        return img, original_shape

    def format_output(self, batch, output):
        batch_boxes, batch_scores = output
        for index in range(batch['image'].size(0)):
            original_shape = batch['shape'][index]
            filename = batch['filename'][index]
            result_file_name = 'res_' + filename.split('/')[-1].split('.')[0] + '.txt'
            result_file_path = os.path.join(self.args['result_dir'], result_file_name)
            boxes = batch_boxes[index]
            scores = batch_scores[index]
            if self.args['polygon']:
                with open(result_file_path, 'wt') as res:
                    for i, box in enumerate(boxes):
                        box = np.array(box).reshape(-1).tolist()
                        result = ",".join([str(int(x)) for x in box])
                        score = scores[i]
                        res.write(result + ',' + str(score) + "\n")
            else:
                with open(result_file_path, 'wt') as res:
                    for i in range(boxes.shape[0]):
                        score = scores[i]
                        if score < self.args['box_thresh']:
                            continue
                        box = boxes[i, :, :].reshape(-1).tolist()
                        result = ",".join([str(int(x)) for x in box])
                        res.write(result + ',' + str(score) + "\n")

    def inference(self, image_list, visualize=False, save=True):
        self.init_torch_tensor()
        model = self.init_model()
        self.resume(model, self.model_path)
        all_matircs = {}

        batch = dict()
        outputs = []
        for i in image_list:
            # batch['filename'] = [i]
            # batch['filename'] = i.filename
            batch['filename'] = i.filename
            img, original_shape = self.load_image(i)
            batch['shape'] = [original_shape]
            model.eval()
            with torch.no_grad():
                batch['image'] = img
                pred = model.forward(batch, training=False)
                output = self.structure.representer.represent(batch, pred, is_output_polygon=self.args['polygon'])
                batch_boxes, _ = output
                outputs.append(batch_boxes)
                if save:
                    if not os.path.isdir(self.args['result_dir']):
                        os.mkdir(self.args['result_dir'])
                    self.format_output(batch, output)
                if visualize and self.structure.visualizer:
                    vis_image = self.structure.visualizer.demo_visualize(i, output)
                    cv2.imwrite(os.path.join(self.args['result_dir'], i.filename.split('/')[-1].split('.')[0] + '.jpg'),
                                vis_image)
        return outputs
