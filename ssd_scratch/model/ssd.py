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
            matched_gt_boxes_per_image = targets_per_image["boxes"] [foreground_matched_idxs_per_image]   # 200 i think, from config, not sure
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
                (cls_logits_per_image.size(0), ),
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
        
        # Hard Negative Mining
        foreground_idxs = cls_targets > 0
        # We will sample total of 3 x (number of fg default_boxes)
        # background default_boxes
        num_negative = self.neg_pos_ratio * foreground_idxs.sum(1, keepdim=True)
        
        
        # As of now cls_loss is for ALL default_boxes
        negative_loss = cls_loss.clone()
        # We want to ensure that after sorting based on loss value, foreground default_boxes are never picked
        # when choosing topK highest loss indexes
        negative_loss[foreground_idxs] = -float("inf")
        values, idx = negative_loss.sort(dim = 1, descending=True)
        
        # Fetch those indexes which have in topK(K = num_negative) losses
        background_idxs = idx.sort(1)[1] < num_negative       
        
        N = max(1, num_foreground)
        return {
            "bbox_regression": bbox_loss.sum() / N,
            "classification": (cls_loss[foreground_idxs].sum() + cls_loss[background_idxs].sum()) / N,
        } 
        
    
    def forward(self, x, targets = None):
        
        # Call everything till conv4_3 layers first
        conv_4_3_out = self.features(x)
        
        # Scale conv4_3 output using learnt norm scale
        conv_4_3_out_scaled = (self.scale_weight.view(1, -1, 1, 1) * torch.nn.functional.normalize(conv_4_3_out))
        
        # Call conv5_3 with non_scaled conv_3 and also call additional conv layers
        conv_5_3_fc_out = self.conv5_3_fc(conv_4_3_out)
        conv8_2_out = self.conv8_2(conv_5_3_fc_out)
        conv9_2_out = self.conv9_2(conv8_2_out)
        conv10_2_out = self.conv10_2(conv9_2_out)
        conv11_2_out = self.conv11_2(conv10_2_out)
        
        # Feature maps for prediction
        outputs = [
            conv_4_3_out_scaled, # 38x38
            conv_5_3_fc_out,     # 19x19
            conv8_2_out,         # 10x10
            conv9_2_out,         # 5x5
            conv10_2_out,        # 3x3
            conv11_2_out         # 1x1
        ]
        
        # Classification and bbox regression for all feature maps
        cls_logits = []
        bbox_reg_deltas = []
        for i, features in enumerate(outputs):
            cls_feat_i = self.cls_heads[i][features]
            bbox_reg_feat_i = self.bbox_reg_heads[i][features]
            
            # Cls output from (B, A * num_classes, H, W) to (B, HWA, num_classes)
            N, _, H, W = cls_feat_i.shape
            cls_feat_i = cls_feat_i.view(N, -1, self.num_classes, H, W)
            # (B, A, num_classes, H, W)
            cls_feat_i = cls_feat_i.permute(0, 3, 4, 1, 2)  # (B, H, W, A, num_classes)
            cls_feat_i = cls_feat_i.reshape(N, -1, self.num_classes)
            
            # (B, HWA, num_classes)
            cls_logits.append(cls_feat_i)
            
            # Permute bbox reg output from (B, A * 4, H, W) to (B, HWA, 4)
            N, _, H, W = bbox_reg_feat_i.shape
            bbox_reg_feat_i = bbox_reg_feat_i.view(N, -1, 4, H, W) # (B, A, 4, H, W)
            bbox_reg_feat_i = bbox_reg_feat_i.permute(0, 3, 4, 1, 2) # (B, H, W, A, 4)
            bbox_reg_feat_i = bbox_reg_feat_i.reshape(N, -1, 4) # size = (B, HWA, 4)
            bbox_reg_deltas.append(bbox_reg_feat_i)
            
        
        # Concat cls logits and bbox regression predictions for all feature maps
        cls_logits = torch.cat(cls_logits, dim=1) # (B, 8732, num_classes)
        bbox_reg_deltas = torch.cat(bbox_reg_deltas, dim= 1) # (B, 8732, 4)
        
        # Generate default_boxes for all feature maps
        default_boxes = generate_default_boxes(outputs, self.aspect_ratios, self.scales)
        # default_boxes -> List[Tensor of shape 8732 x 4]
        # len(default_boxes) = batch_size
        
        losses = []
        detections  =[]
        if self.training:
            # List to hold for each image, which default box is assigned to with gt box if any or unassigned(background)
            matched_idxs = []
            for default_boxes_per_image, targets_per_image in zip(default_boxes, targets):
                
                if targets_per_image["boxes"].numel == 0:
                    matched_idxs.append(
                        torch.full(
                            (default_boxes_per_image.size(0), 1), -1,
                            dtype = torch.int64,
                            device = default_boxes_per_image.device
                        )
                    )        
                    
                    continue
                iou_matrix = get_iou(targets_per_image["boxes"], default_boxes_per_image)  
                # For each default box find best ground truth box
                matched_vals, matches = iou_matrix.max(dim = 0) 
                # matches -> [8732]
                
                # Update index of match for all default_boxes which have maximum iou with a gt box < low_threshold as -1
                # This allows selecting foreground boxes as match_index >= 0
                below_low_threshold = matched_vals < self.iou_threshold
                matches[below_low_threshold] =-1
                
                # We want to also assign the best default box for every gt as foreground
                # So first find the best default box for every gt
                _, highest_quality_pred_foreach_gt = iou_matrix.max(dim = 1)
                # Update the best matching gt index for these best default_boxes
                # as 0, 1, 2, ...., len(gt) - 1
                matches[highest_quality_pred_foreach_gt] = torch.arange(
                    highest_quality_pred_foreach_gt.size(0), dtype=torch.int64,
                    device=highest_quality_pred_foreach_gt.device
                )
                
                matched_idxs.append(matches)
                
            losses = self.compute_loss(targets, cls_logits, bbox_reg_deltas, default_boxes, matched_idxs)
        
        else:
            # For test time we do the following:
            # 1. Convert default_boxes to boxes using predicted bbox regression deltas
            # 2. Low score filtering
            # 3. Pre-NMS TopK filtering
            # 4. NMS
            # 5. Post NMS TopK filtering
            cls_scores = torch.nn.functional.softmax(cls_logits, dim=-1)
            num_classes = cls_scores.size(-1)
            
            for bbox_deltas_i, cls_scores_i, default_boxes_i in zip(bbox_reg_deltas, cls_scores, default_boxes):
                boxes = apply_regression_pred_to_default_boxes(bbox_deltas_i, default_boxes_i)
                
                # Ensure all values are between 0 and 1
                boxes.clamp_(min = 0., max = 1.0)
                
                pred_boxes = []
                pred_scores = []
                pred_labels = []
                # Class wise filtering
                for label in range(1, num_classes):
                    score = cls_scores_i[:, label]
                    
                    # Remove low scoring boxes of this class
                    keep_idxs = score > self.low_score_threshold
                    score = score[keep_idxs]
                    box = boxes[keep_idxs]
                    
                    # keep only topk scoring predictions of this class
                    score, top_k_idxs = score.topk(min(self.pre_nms_topK, len(score)))
                    box = box[top_k_idxs]
                    
                    pred_boxes.append(box)
                    pred_scores.append(score)
                    pred_labels.append(torch.full_like(score, fill_value=label, dtype = torch.int64, device=cls_scores.device))
            
            pred_boxes = torch.cat(pred_boxes, dim=0)
            pred_scores = torch.cat(pred_scores, dim=0)
            pred_labels = torch.cat(pred_labels, dim=0)
            
            # Class wise NMS
            keep_mask = torch.zeros_like(pred_scores, dtype=torch.bool)
            for class_id in torch.unique(pred_labels):
                curr_indices = torch.where(pred_labels == class_id)[0]
                curr_keep_idxs = torch.ops.torchvision.nms(pred_boxes[curr_indices], pred_scores[curr_indices], self.nms_threshold)
                
                keep_mask[curr_indices[curr_keep_idxs]] = True
            
            keep_indices = torch.where(keep_mask)[0]
            post_nms_keep_indices = keep_indices[pred_scores[keep_indices].sort(descending=True)[1]]
            keep = post_nms_keep_indices[:self.detections_per_img]
            pred_boxes, pred_scores, pred_labels = (pred_boxes[keep], pred_scores[keep], pred_labels[keep])
            
            detections.append(
                {
                    "boxes": pred_boxes,
                    "scores": pred_scores,
                    "labels": pred_labels
                }
            )
            
        return losses, detections
                                
                
        
        
        
        
        
            
            
            
              
            

        
        
        
        
        
        
        
        
        
        
        
        
    