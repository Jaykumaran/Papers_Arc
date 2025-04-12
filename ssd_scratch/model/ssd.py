import torch
import torch.nn as nn
import math
import torchvision



def get_iou(boxes1, boxes2):
    """
    IoU between two set of boxes

    :param boxes1: (Tensor of shape N x 4)
    :param boxes2: (Tensor of shape M x 4)
    
    return IoU matrix of shape N x M
    
    
    For eg:
   boxes1 = torch.tensor([
        [1, 1, 4, 4],   # Box A (x1, y1, x2, y2)
        [2, 2, 5, 5]    # Box B
    ])  # Shape (2, 4)

    boxes2 = torch.tensor([
        [3, 3, 6, 6],   # Box C
        [0, 0, 2, 2]    # Box D
    ])  # Shape (2, 4)
 
    """
    
    # Area of boxes (x2-x1)*(y2-y1)
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1]) # (N, ) ; 
    # boxes1[:, 2] - boxes1[:, 0] → [4 - 1, 5 - 2] = [3, 3]   # Matrix Subtraction
    # boxes1[:, 3] - boxes1[:, 1] → [4 - 1, 5 - 2] = [3, 3]
    # area1 = [3 * 3, 3 * 3] = [9, 9] # Shape (2,)

    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1]) # (M, ) ; 
    # boxes2[:, 2] - boxes2[:, 0] → [6 - 3, 2 - 0] = [3, 2]
    # boxes2[:, 3] - boxes2[:, 1] → [6 - 3, 2 - 0] = [3, 2]
    # area2 = [3 * 3, 2 * 2] = [9, 4] # Shape (2,)


    # Get top left x1,y1 coordinate
    x_left = torch.max(boxes1[:, None, 0], boxes2[None, :, 0])   # Broadcast 
    # boxes1[:, None, 0] → [[1], [2]] → shape (2,1)
    # boxes2[:, 0] → [3, 0] → shape (2,)

    y_top = torch.max(boxes1[:, None, 1], boxes2[None, : , 1])  # (N, M)
    
    # Get bottom right x2, y2 coordinate
    x_right = torch.min(boxes1[:, None, 2], boxes2[None, :, 2]) # (N, M)
    # x_right - x_left = [[4 - 3, 2 - 1], 
                   #      [5 - 3, 2 - 2]] = [[1, 1], [2, 0]]
    y_bottom = torch.max(boxes1[:, None, 3], boxes2[None, :, 3]) # (N, M)
    
    intersection_area = ((x_right - x_left).clamp(min = 0) * (y_bottom - y_top).clamp(min = 0)) # (N, M)
    
    
    
    
class SSD(nn.Module):
    
    """
    Main class for SSD. Does the following steps to generate detections/losses
    During initialization
    1. Load VGG Imagenet pretrained model
    2. Extract Backbone from VGG and add extra conv layers
    3. Add class prediction and bbox transformation prediction layers
    4. Initialize all layers
    
    During Forward pass:
    1. Get conv4_3 output
    2. Normalize and scale conv4_3 output (feat_output_1)
    3. Pass the unscaled conv4_3 to conv5_3 layers and conv layers
        replacing fc6 and fc7 of vgg (feat_output_2)
    4. Pass the conv_fc7 output to extra conv layers (feat_output_3-6)
    5. Get the classification and regression predictions for all 6 feature maps
    6. Generate feature boxes for all these feature maps (8732 x 4)
    7a. If in training assign targets for these default_boxes and compute localization
        and classification losses
    7b. If in inference mode, then do all pre-nms filtering, nms and then post nms filtering
        and return the detected boxes, their labels and their scores.
    
    """
    
    def __init__(self, config, num_classes = 21): # pascal voc
        super().__init__()
        
        self.aspect_ratios = config['aspect_ratios']
        
        self.scales = config['scales']
        self.scales.append(1.0)
        
        self.num_classes = num_classes
        self.iou_threshold = config['iou_threshold']
        self.low_score_threshold = config['low_score_threshold']
        self.neg_pos_ratio = config['neg_pos_ratio']
        self.pre_nms_topK = config['pre_nms_topK']
        self.nms_threshold = config['nms_threshold']
        self.detections_per_img = config['detections_per_img']  # Post nms how many detections to keep per image.
        
        # Load imagenet pretrained weights
        backbone = torchvision.models.vgg16(
            weights = torchvision.models.VGG16_Weights.IMAGENET1K_V1
        )
        
        # Get all max pool indexes to determine different stages
        max_pool_pos = [idx for idx, layer in enumerate(list(backbone.features)) if isinstance(layer, nn.MaxPool2d)]  
        
        max_pool_stage_3_pos = max_pool_pos[-3]
        max_pool_stage_4_pos = max_pool_pos[-2]
        
        backbone.features[max_pool_stage_3_pos].ceil_mode = True
        # otherwise vgg conv4_3 output will be 37x37
        self.features = nn.Sequential(*backbone.features[:max_pool_stage_4_pos])
        self.scale_weight = nn.Parameter(torch.ones(512) * 20)
        
        #########################################
        # Conv5_3 + Conv for fc6 and fc 7
        #########################################
        # Conv modules replacing fc6 and fc7
        # Ideally we would copy the weights
        # but here we are just adding new layers
        # and not copying fc6 and fc7 weights by 
        # subsampling
        
        fcs = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=6, dilation=6),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        
        self.conv5_3_fc = nn.Sequential(
            *backbone.features[max_pool_stage_4_pos:-1],
            fcs
        )
        
        ###############################################
        # Additional conv layers
        ###############################################
        # Modules to take from 19x19 to 10x10
        self.conv8_2 = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=1, stride=1),
            nn.MaxPool2d(inplace = True),
            nn.Conv2d(256, 512, kernel_size=3, stride=2),
            nn.MaxPool2d(inplace = True)
        )
        
        # Modules to take from 10x10 to 5x5
        self.conv9_2 = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=1, stride=1),
            nn.MaxPool2d(inplace = True),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2),
            nn.ReLU(inplace=True)
        )
        
        
        # Modules to take from 5x5 to 3x3
        self.conv10_2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1, stride=1),
            nn.MaxPool2d(inplace = True),
            nn.Conv2d(128, 256, kernel_size=3),
            nn.ReLU(inplace=True)
        )
        
        # Modules to take from 3x3 to 1x1
        self.conv11_2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3),
            nn.ReLU(inplace=True)
        )
        
        
        # Must match con4_3, fcs, conv8_2, conv9_2, conv10_2, conv11_2
        out_channels = [512, 1024, 512, 256, 256, 256]
        
        
        ########################################
        # Prediction layers 
        ########################################
        # Classification head
        self.cls_heads = nn.ModuleList()
        for channels, aspect_ratio in zip(out_channels, self.aspect_ratios):
            # extra 1.0 is added for scale of sqrt(sk*sk+1)
            self.cls_heads.append(nn.Conv2d(channels, 
                                            self.num_classes * (len(aspect_ratio) + 1),
                                            kernel_size=1,
                                            padding=1
                                            ))
            
        # Box Head
        self.bbox_reg_heads = nn.ModuleList()
        for channels, aspect_ratio in zip(out_channels, self.aspect_ratios):
            # extra 1.0 is added for scale of sqrt(sk*sk+1)
            self.bbox_reg_heads.append(nn.Conv2d(channels, 4 * (len(aspect_ratio) +1),
                                                 kernel_size=3,
                                                 padding=1))
            
        ########################################
        # Conv Layer Initialization # To get stable diverse gradient maps across features
        ########################################
        for layer in fcs.modules:
            if isinstance(layer, nn.Conv2d):
                torch.nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    torch.nn.init.constant_(layer.bias, 0.0)
        
        for module in self.cls_heads:
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0.0)
        
        for module in self.bbox_reg_heads:
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0.0) # set the const value as 0.0
                
                
    def compute_loss(self, targets, cls_logits, bbox_regression, default_boxes, matched_idxs):
        
        # Counting all the foreground default boxes for computing N in loss equation
        num_foreground = 0
        # BBox losses for all batch images(for foreground default boxes)
        bbox_loss = []
        # classification targets for all batch images(for ALL default boxes)
        cls_targets = []
        for (
            targets_per_image,
            bbox_regression_per_image,
            cls_logits_per_image,
            default_boxes_per_image,
            matched_idxs_per_image,  # match anchors with gt for either fg or bg (-1)
        ) in zip(targets, bbox_regression, cls_logits, default_boxes, matched_idxs):
            # Foreground default boxes -> matched_idx >= 0
            # Background default boxes -> matched_idx = -1
            fg_idxs_per_image = torch.where(matched_idxs_per_image >= 0)[0]
            foreground_matched_idxs_per_image = matched_idxs_per_image[fg_idxs_per_image]
            
            num_foreground += foreground_matched_idxs_per_image.numel()
            
            # Get foreground default boxes and their transformation predictions
            matched_gt_boxes_per_image = targets_per_image["boxes"]   # 200 i think, from config, not sure
            bbox_regression_per_image = bbox_regression_per_image[fg_idxs_per_image, :]
            target_regression  = boxes_to_transformation_targets(
                matched_gt_boxes_per_image,
                default_boxes_per_image
            )
            
            
            bbox_loss.append(
                torch.nn.functional.smooth_l1_loss(bbox_regression_per_image,
                                                   target_regression,
                                                   reduction='sum')
            )
            
            # Get classification target for ALL default boxes
            # For all default boxes set it at 0 first
            # Then set foreground default boxes target as label
            # of assigned gt box
            gt_classes_target = torch.zeros(
                (cls_logits_per_image.size(0), )
                dtype = targets_per_image["labels"].dtype,
                device = targets_per_image["labels"].device,
            )
            
            
            gt_classes_target[fg_idxs_per_image] = targets_per_image["labels"][foreground_matched_idxs_per_image]
            
            cls_targets.append(gt_classes_target)
            
        # Aggregated bbox loss and classification targets for all batch images
        bbox_loss = torch.stack(bbox_loss)
        cls_targets = torch.stack(cls_targets) # (B, 8732)
        
        # Calculate classification loss for ALL default_boxes
        num_classes = cls_logits.size(-1)
        cls_loss = torch.nn.functional.cross_entropy(cls_logits.view(-1, num_classes), cls_targets.view(-1), reduction="none").view(cls_targets.size())
        
        
        
        
            
            
            
              
            

        
        
        
        
        
        
        
        
        
        
        
        
    