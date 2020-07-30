import numpy as np
import math
import cv2


def adjust_bbox(bbox, reference_bbox):
    new_origin_x, new_origin_y = reference_bbox[0], reference_bbox[1]
    adjusted_bbox = [bbox[0] - new_origin_x, bbox[1] - new_origin_y, bbox[2] - new_origin_x, bbox[3] - new_origin_y]
    return adjusted_bbox


# from https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
def iou(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
 
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
 
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
 
    # return the intersection over union value
    return iou


# if tube width< 128 and height<72 then 
def splating_bbox(tube, threshold, frames_input_height, frames_input_width):
    width_tube = tube[2] - tube[0] + 1
    height_tube = tube[3] - tube[1] + 1


    if width_tube < threshold:                      
        width_offset = threshold - width_tube       
        last_half = math.ceil(width_offset/2)
        first_half = math.floor(width_offset/2)
        if tube[2] + last_half < frames_input_width:  
            if tube[0] - first_half >=0:  
                tube[2] = tube[2] + last_half
                tube[0] = tube[0] - first_half
            else:
                tube[2] = tube[2] + width_offset
                
        else:                                                     
            tube[0] = tube[0] - width_offset

    if height_tube < threshold:
        height_offset = threshold - height_tube
        last_half = math.ceil(height_offset/2)
        first_half = math.floor(height_offset/2)

        if tube[3] + last_half < frames_input_height:
            if tube[1] - first_half >= 0:
                tube[3] = tube[3] + last_half
                tube[1] = tube[1] - first_half
            else:
                tube[3] = tube[3] + height_offset        
        else:
            tube[1] = tube[1] - height_offset

    return tube

#############unused currently
# if tube width< 128 and height<72 then 
# tube_h or tube_w < 112 ....make it 112x112.. if it is >112 say, 200x500... make it square 
def check_min_criteria(tube, frames_input_height, frames_input_width, min_threshold = 96):


    width_tube = tube[2] - tube[0] + 1
    height_tube = tube[3] - tube[1] + 1

    if width_tube < min_threshold and height_tube < min_threshold:
        threshold = min_threshold
        tube = splating_bbox(tube, threshold, frames_input_height, frames_input_width)
        
    else:
        threshold = max(width_tube, height_tube)
        tube = splating_bbox(tube, threshold, frames_input_height, frames_input_width)
        
    '''
    elif width_tube < min_threshold or height_tube < min_threshold:
        threshold = max(width_tube, height_tube)
        tube = splating_bbox(tube, threshold, frames_input_height, frames_input_width)    
    '''
    
    return tube 



# boxA - ground_truth bbox, boxB - bbox we are cropping
def get_bbox_overlap(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the ground_truth bbox and final crop bbox 
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    #if boxAArea == 0:
    #   print(boxA, ' has area 0')
    #   return 0

    # compute the intersection of ground_truth bbox with the bbox we are cropping
    overlap = interArea / float(boxAArea)

    # return the overlap
    return overlap




def sliding_window(arr, size, stride):
    num_chunks = int((len(arr) - size) / stride) + 2
    result = []
    for i in range(0,  num_chunks * stride, stride):
        if len(arr[i:i + size]) > 0:
            result.append(arr[i:i + size])
    return np.array(result)


def adjust_boundaries(x, x_max):
    if x not in range(0, x_max):
        if x < 0:
            return 0
        else:
            return x_max
    return x


def adjust_bboxes(bboxes, size, frame_width, frame_height):
    
    adjusted_bboxes = []
    for bbox in bboxes:
        x_min = int((bbox[0] + bbox[2] - size) / 2)
        y_min = int((bbox[1] + bbox[3] - size) / 2)
        x_max = int((bbox[0] + bbox[2] + size) / 2)
        y_max = int((bbox[1] + bbox[3] + size) / 2)
        bbox_square =  [x_min, y_min, x_max, y_max]
        
        bbox_square = [int(math.ceil(i)) for i in bbox_square]
        diff = (bbox_square[2] - bbox_square[0]) - (bbox_square[3] - bbox_square[1])
        if diff > 5:
            print("Bounding box adjusted by more then 5 pixels")

        if diff > 0:
            bbox_square[2] -= diff
        if diff < 0:
            bbox_square[3] += diff
        diff = (bbox_square[2] - bbox_square[0]) - (bbox_square[3] - bbox_square[1])
        assert diff == 0
     
        assert bbox_square[2] - bbox_square[0] == size
        assert bbox_square[3] - bbox_square[1] == size

        if (x_min >= 0 and x_min <= frame_width) and (y_min >= 0 and y_min <= frame_height) and (x_max >= 0 and x_max <= frame_width) and (y_max >= 0 and y_max <= frame_height):
            adjusted_bboxes.append(bbox_square)
        else:
           # handle different cases 
            adjusted_bboxes.append(None)
    return adjusted_bboxes


def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]


# scale factor is the context we adding (default - 0.5 i.e., 50% extra)
def reshape_bbox_multiscale(bbox, scale_factor, min_size, frame_width, frame_height):
    assert type(bbox) is list
    assert len(bbox) == 4

    bbox_width = bbox[2] - bbox[0]
    bbox_height = bbox[3] - bbox[1]
    size = max(bbox_width, bbox_height)
        
    # important part of the code - adjusts the size of the crop depending on the original size
    if size < min_size/2:
       size = int(min_size/2) 
    elif size < min_size:
        size = min_size
    
    x_min = int((bbox[0] + bbox[2] - size) / 2)
    y_min = int((bbox[1] + bbox[3] - size) / 2) 
    x_max = int((bbox[0] + bbox[2] + size) / 2)
    y_max = int((bbox[1] + bbox[3] + size) / 2)
    bbox_square =  [x_min, y_min, x_max, y_max]

    bbox_width = bbox_square[2] - bbox_square[0]
    bbox_height = bbox_square[3] - bbox_square[1]
    x_min = bbox_square[0] - ((scale_factor / 2) * bbox_width)
    y_min = bbox_square[1] - ((scale_factor / 2) * bbox_height)
    x_max = bbox_square[2] + ((scale_factor / 2) * bbox_width)
    y_max = bbox_square[3] + ((scale_factor / 2) * bbox_height)
    
    bbox_new = [x_min, y_min, x_max, y_max]

    final_bbox, tag = None, None    
    if (x_min >= 0 and y_min >= 0) and (x_max <= frame_width and y_max <= frame_height):
        final_bbox, tag = [x_min, y_min, x_max, y_max], 'center'
   
    if (x_min <= 0 and y_min <= 0) and (x_max <= frame_width and y_max <= frame_height):
        final_bbox, tag = [0, 0, x_max - x_min, y_max - y_min], 'left_top'
    
    if (x_min <= 0 and y_min >= 0) and (x_max <= frame_width and y_max <= frame_height):
        final_bbox, tag = [0, y_min, x_max - x_min, y_max], 'left'
    
    if (x_min <= 0 and y_min >= 0) and (x_max <= frame_width and y_max >= frame_height):
        final_bbox, tag = [0, y_min - y_max + frame_height, x_max - x_min, frame_height], 'left_bottom'
    
    if (x_min >= 0 and y_min <= 0) and (x_max <= frame_width and y_max <= frame_height):
        final_bbox, tag = [x_min, 0, x_max, y_max - y_min], 'top'
    
    if (x_min >= 0 and y_min >= 0) and (x_max <= frame_width and y_max >= frame_height):
        final_bbox, tag = [x_min, y_min - y_max + frame_height, x_max, frame_height], 'bottom'
    
    if (x_min >= 0 and y_min <= 0) and (x_max >= frame_width and y_max <= frame_height):
        final_bbox, tag = [x_min - x_max + frame_width, 0, frame_width, y_max - y_min], 'right_top'
    
    if (x_min >= 0 and y_min >= 0) and (x_max >= frame_width and y_max <= frame_height):
        final_bbox, tag = [x_min - x_max + frame_width, y_min, frame_width, y_max], 'right'
    
    if (x_min >= 0 and y_min >= 0) and (x_max >= frame_width and y_max >= frame_height):
        final_bbox, tag = [x_min - x_max + frame_width, y_min - y_max + frame_height, frame_width, frame_height], 'right_bottom'    
    
    if final_bbox is None or tag is None:
        return [-1, -1, -1, -1], None 
    
    final_bbox = [int(math.ceil(i)) for i in final_bbox]
    diff = (final_bbox[2] - final_bbox[0]) - (final_bbox[3] - final_bbox[1])
    if diff > 5:
        print("Bounding box adjusted by more then 5 pixels")

    if diff > 0:
        final_bbox[2] -= diff      
    if diff < 0:
        final_bbox[3] += diff   
    diff = (final_bbox[2] - final_bbox[0]) - (final_bbox[3] - final_bbox[1])
    
    assert diff == 0  
    
    if final_bbox[0] < 0 or final_bbox[1] < 0 or final_bbox[2] < 0 or final_bbox[3] < 0:
        print(final_bbox, bbox_new, frame_width, frame_height)

    return final_bbox, tag


def reshape_bbox(bbox, size, x_max, y_max):
    assert type(bbox) is list
    assert len(bbox) == 4
    if size is None:
        size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
    else:
        size = max(size, max(bbox[2] - bbox[0], bbox[3] - bbox[1]))
    x1 = adjust_boundaries(int((bbox[0] + bbox[2] - size) / 2), x_max)
    y1 = adjust_boundaries(int((bbox[1] + bbox[3] - size) / 2), y_max)
    x2 = adjust_boundaries(int((bbox[0] + bbox[2] + size) / 2), x_max)
    y2 = adjust_boundaries(int((bbox[1] + bbox[3] + size) / 2), y_max)
    return [x1, y1, x2, y2]


def rescale_frame(frame_height, frame_width, min_size, max_size):
    
    #aspect_ratio = float(frame_height)/frame_width
   
    final_scale_factor = 0

    min_dim = min(frame_height, frame_width)
    max_dim = max(frame_height, frame_width)

    if min_dim == max_dim:
        scale_factor = float(max_size)/min_dim
        scaled_height = int(math.ceil(frame_height * scale_factor))
        scaled_width = int(math.ceil(frame_width * scale_factor))
        final_scale_factor = scale_factor
    else:
        if min_dim < min_size:
            scale_factor = float(min_size)/min_dim
            scaled_height = int(math.ceil(frame_height * scale_factor))
            scaled_width = int(math.ceil(frame_width * scale_factor))
            final_scale_factor = scale_factor
            if max(scaled_height, scaled_width) < max_size:
                  new_scale_factor = float(max_size)/max(scaled_height, scaled_width)
                  scaled_height = int(math.ceil(scaled_height * new_scale_factor))
                  scaled_width = int(math.ceil(scaled_width * new_scale_factor))
                  final_scale_factor = new_scale_factor
        else:
            if max_dim > max_size:
                scale_factor = float(max_size)/max_dim
                scaled_height = int(math.ceil(frame_height * scale_factor))
                scaled_width = int(math.ceil(frame_width * scale_factor))
                final_scale_factor = scale_factor
                if min(scaled_height, scaled_width) < min_size:
                    new_scale_factor = float(min_size)/min(scaled_height, scaled_width)
                    scaled_height = int(math.ceil(scaled_height * new_scale_factor))
                    scaled_width = int(math.ceil(scaled_width * new_scale_factor))
                    final_scale_factor = new_scale_factor
            else:
                scale_factor = float(max_size)/max_dim
                scaled_height = int(math.ceil(frame_height * scale_factor))
                scaled_width = int(math.ceil(frame_width * scale_factor))
                final_scale_factor = scale_factor

    assert min(scaled_height, scaled_width) >= min_size
    
    new_aspect_ratio = float(scaled_height)/scaled_width
    
    return scaled_height, scaled_width, final_scale_factor


def get_rescale_factor(clip_width, clip_height, min_size, max_size):
    min_dim = min(clip_height, clip_width)
    max_dim = max(clip_height, clip_width)

    if min_dim == max_dim:
        scale_factor = float(max_size)/min_dim
        scaled_height = int(math.ceil(clip_height * scale_factor))
        scaled_width = int(math.ceil(clip_width * scale_factor))
    else:
        if min_dim < min_size:
            scale_factor = float(min_size)/min_dim
            scaled_height = int(math.ceil(clip_height * scale_factor))
            scaled_width = int(math.ceil(clip_width * scale_factor))
            if max(scaled_height, scaled_width) < max_size:
                  scale_factor = float(max_size)/max(scaled_height, scaled_width)
                  scaled_height = int(math.ceil(scaled_height * scale_factor))
                  scaled_width = int(math.ceil(scaled_width * scale_factor))
        else:
            if max_dim > max_size:
                scale_factor = float(max_size)/max_dim
                scaled_height = int(math.ceil(clip_height * scale_factor))
                scaled_width = int(math.ceil(clip_width * scale_factor))
                if min(scaled_height, scaled_width) < min_size:
                    scale_factor = float(min_size)/min(scaled_height, scaled_width)
                    scaled_height = int(math.ceil(scaled_height * scale_factor))
                    scaled_width = int(math.ceil(scaled_width * scale_factor))
            else:
                scale_factor = float(max_size)/max_dim
                scaled_height = int(math.ceil(clip_height * scale_factor))
                scaled_width = int(math.ceil(clip_width * scale_factor))

    assert min(scaled_height, scaled_width) >= min_size

    return scale_factor


def rescale_crop(cropped_regions, min_size, max_size):
    clip_shape = np.array(cropped_regions).shape
    if len(clip_shape) < 3:
        return None
    clip_height = clip_shape[1]
    clip_width = clip_shape[2]
    min_dim = min(clip_height, clip_width)
    max_dim = max(clip_height, clip_width)

    if min_dim == max_dim:
        scale_factor = float(max_size)/min_dim
        scaled_height = int(math.ceil(clip_height * scale_factor))
        scaled_width = int(math.ceil(clip_width * scale_factor))
    else:
        if min_dim < min_size:
            scale_factor = float(min_size)/min_dim
            scaled_height = int(math.ceil(clip_height * scale_factor))
            scaled_width = int(math.ceil(clip_width * scale_factor))
            if max(scaled_height, scaled_width) < max_size:
                  new_scale_factor = float(max_size)/max(scaled_height, scaled_width)
                  scaled_height = int(math.ceil(scaled_height * new_scale_factor))
                  scaled_width = int(math.ceil(scaled_width * new_scale_factor))
        else:
            if max_dim > max_size:
                scale_factor = float(max_size)/max_dim
                scaled_height = int(math.ceil(clip_height * scale_factor))
                scaled_width = int(math.ceil(clip_width * scale_factor))
                if min(scaled_height, scaled_width) < min_size:
                    new_scale_factor = float(min_size)/min(scaled_height, scaled_width)
                    scaled_height = int(math.ceil(scaled_height * new_scale_factor))
                    scaled_width = int(math.ceil(scaled_width * new_scale_factor))
            else:
                scale_factor = float(max_size)/max_dim
                scaled_height = int(math.ceil(clip_height * scale_factor))
                scaled_width = int(math.ceil(clip_width * scale_factor))

    assert min(scaled_height, scaled_width) >= min_size

    rescaled_crop = []
    for region in cropped_regions:
       rescaled_frame = cv2.resize(region, (scaled_height, scaled_width))
       rescaled_crop.append(rescaled_frame)
    return rescaled_crop


def load_model(saved_model_file, num_classes, model):
    #model = video_models.r2plus1d_18(pretrained=False, progress=False)
    #model.fc = nn.Linear(512, num_classes)
    #model = nn.Sequential(model, nn.Sigmoid())

    pretrained = torch.load(saved_model_file)
    pretrained_kv_pair = pretrained['state_dict']
    
    model_kv_pair = model.state_dict()
    for layer_name, weights in pretrained_kvpair.items():
        layer_name = layer_name.replace('module.','')
        
        if 'fc.weight' in layer_name or 'fc.bias' in layer_name:
            print('skipped weight loading for layer: ',layer_name)
            continue
        
        model_kvpair[layer_name]=weights
    model.load_state_dict(model_kvpair, strict=False)
    return model


