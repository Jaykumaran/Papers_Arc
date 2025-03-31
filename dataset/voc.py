# Custom Dataset Preparation
import os
import torch
import torchvision.transforms.v2 as v2
from torch.utils.data import Dataset
import xml.etree.ElementTree as ET
from torchvision import tv_tensors
from torchvision.io import read_image
from typing import Dict


# /home/jaykumaran/Downloads/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007
# ├── Annotations
# ├── ImageSets
# │   ├── Layout
# │   ├── Main
# │   └── Segmentation
# ├── JPEGImages
# ├── SegmentationClass
# └── SegmentationObject

def load_images_and_anns(im_sets: str, label2idx: Dict, ann_filename: str, split: str = None):
    
    r"""
    Method to get the xml files and for each file get all the objects and their ground truth bboxes 
    :param im_sets: Sets of images to consider
    :param label2idx: Class name to index mapping
    :param ann_name: txt file containing image names{trainval.txt or test.txt}
    """
    
    im_infos = []
    ims = []
    
    
    for im_set in im_sets:
        im_names = []
        # Fetch all image names in txt file for this imageset
        for line in open(os.path.join(im_set, 'ImageSets', 'Main', f'{ann_filename}.txt')):  # Eg: aeroplane_train.txt
            im_names.append(line.strip())
            
        # Set annotaton and image path
        ann_dir = os.path.join(im_set, 'Annotations') 
        im_dir = os.path.join(im_set, 'JPEGImages')
        
        for im_name in im_names:
            ann_file = os.path.join(ann_dir, f'{im_name}.xml') # 000005.xml
            
            if not os.path.exists(ann_file):
                print(f"Warning: Annotation file for {ann_file} not found")
                continue
            im_info = {}
            ann_info = ET.parse(ann_file)
            root = ann_info.getroot() 
            
            # <annotation><folder>VOC2007</folder><filename>000005.jpg</filename> <size> <segment> <object> <name> <bndbox>. . </annotation>
            
            size = root.find('size') # <size> <width> <height> <depth> <size>
            width = int(size.find('width').text)
            height = int(size.find('height').text)
            im_info['img_id'] = os.path.basename(ann_file).split('.xml')[0] # get image name without extension
            im_info['filename'] = os.path.join(
                im_dir, f"{im_info['img_id']}.jpg"
            )
            im_info['width'] = width
            im_info['height'] = height
            detections = []
            
            for obj in ann_info.findall('object'): # get all occurences of object in a xml file
                det = {}
                label = label2idx[obj.find('name').text]
                difficult = int(obj.find('difficult').text) # 1 for difficult detections either due to occlusion , partial visible
                bbox_info = obj.find('bndbox')
                # Convert 1-based VOC bounding box coordinates to 0-based indexing
                # Pascal deafult xml values start from 1 whereas opencv, pytorch starts at 0 index
                bbox = [
                    int(bbox_info.find('xmin').text) -1,    
                    int(bbox_info.find('ymin').text) -1,
                    int(bbox_info.find('xmax').text) -1,
                    int(bbox_info.find('ymax').text) -1 
                ]
                
                det['label'] = label
                det['difficult'] = difficult
                det['bbox'] = bbox
                
                # At test time eval, ignore difficult
                detections.append(det)
                
                
            im_info['detections'] = detections  # bbox data
            im_infos.append(im_info)  # image metadata
            
    print(f"Total {len(im_infos)} Images found")
    return im_infos
            
            
            
class VOCDataset(Dataset):
    def __init__(self, split, im_sets, im_size = 300):
        
        self.split = split
        
        # Image set for this dataset (VOC2007/VOC2007+VOC2012/VOC2007-test)  
        self.im_sets = im_sets
        self.fname = 'trainval' if self.split == 'train' else 'test'
        
        self.im_size = im_size
        self.im_mean = [123.0, 117.0, 104.0]  # mean pixel values in BGR format --> OpenCV
        self.imagenet_mean = [0.485, 0.456, 0.406] # imagenet
        self.imagenet_std = [0.229, 0.224, 0.225]
        
        
        # 👉 Zoom out Aug is introduced to increase small samples by reducing to 1/4 of input image and place at a random location in the **canvas** (size is 300x300).
        # during training where the pads will have im_mean
        
        # 👉 Random IOU_crop, crops a part of the image while ensuring that the cropped region still contains objects with a certain overlap (IoU) with the original bounding boxes.



        # 👉 Boxes that are outside the IOU crop are ignored using SanitizeBoundingBoxes method. These are boxes that are no longer valid


        # Transformations
        self.transforms = {
            'train': v2.Compose([
                v2.RandomPhotometricDistort(p = 0.5),
                v2.RandomZoomOut(fill = self.im_mean),
                v2.RandomIoUCrop(),  # requires tv_tensors bbox as input
                v2.RandomHorizontalFlip(p=0.5),
                v2.Resize(size = (self.im_size, self.im_size)),
                v2.SanitizeBoundingBoxes(
                    labels_getter=lambda transform_input:
                        (transform_input[1]['labels'], transform_input[1]['difficult'])
                ),
                v2.ToPureTensor(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean = self.imagenet_mean, std = self.imagenet_std)
            ]),
            
            'test': v2.Compose([
                v2.Resize(size = (self.im_size, self.im_size)),
                v2.ToPureTensor(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean = self.imagenet_mean, std = self.imagenet_std) 
            ])
        }
        
        classes = [
            'person',
            'bird',
            'cat',
            'cow',
            'dog',
            'horse',
            'sheep',
            'aeroplane',
            'bicycle',
            'boat',
            'bus',
            'car',
            'motorbike',
            'train',
            'bottle',
            'chair',
            'diningtable',
            'pottedplant',
            'sofa',
            'tvmonitor'
        ]
        
        
        classes = sorted(classes)  # sorts alphabetically
        
        # Add background class at 0 index
        classes = ['background'] + classes
        
        self.label2idx = {classes[idx]: idx for idx in range(len(classes))}
        self.idx2label = {idx: classes[idx] for idx in range(len(classes))}
        
        print(self.idx2label)
        self.images_info = load_images_and_anns(im_sets=self.im_sets,
                                                label2idx=self.label2idx,
                                                ann_filename= self.fname,
                                                split = self.split)
        
    def __len__(self):
        return len(self.images_info)
    
    def __getitem__(self, index):
        im_info = self.images_info[index]
        im = read_image(im_info['filename'])
        
        # Get corresponding annotations for the image
        targets = {}
        targets['bboxes'] = tv_tensors.BoundingBoxes(
            
           [detection['bbox'] for detection in im_info['detections']],
           format='XYXY', canvas_size=im.shape[-2:]
        )
        
        targets['labels'] = torch.as_tensor(
           [detection['label'] for detection in im_info['detections']]
        )
        
        targets['difficult'] = torch.as_tensor(
            [detection['difficult'] for detection in im_info['detections']]
        )
        
        # Transform the image and targets
        transformed_info = self.transforms[self.split](im, targets)
        im_tensor, targets = transformed_info
        
        h, w = im_tensor.shape[-2:]
        # Normalize the bbox to make it invariant
        wh_tensor = torch.as_tensor([[w, h, w, h]]).expand_as(targets['bboxes'])
        targets['bboxes'] = targets['bboxes'] / wh_tensor
        return im_tensor, targets, im_info['filename']
        
        
        
        
        





