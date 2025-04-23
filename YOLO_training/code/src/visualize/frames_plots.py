
from PIL import Image, ImageDraw, ImageFont
import torch
import numpy as np

def get_annotated_frame(frame: torch.Tensor, results, rgb=True):
    # Convert the frame to a PIL Image for easier drawing
    frame = np.repeat(frame[:, :, np.newaxis], 3, axis=2)
    frame = 512*frame/8
    print(frame.shape)
    frame = frame.cpu().byte()  # Ensure the frame is in the byte format
    pil_frame = Image.fromarray(frame.numpy())
    draw = ImageDraw.Draw(pil_frame)

    # Font settings (default font, you can specify a custom TTF if desired)
    try:
        font = ImageFont.load_default()
    except IOError:
        font = None

    if results is not None:
        for res in results:
            print(res)
            [x_min, y_min, x_max, y_max, conf, obj_class] = res
            x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
            obj_class = int(obj_class)

            # Draw the bounding box with color based on the class
            color = (200 * (obj_class == 2), 200 * (obj_class == 1), 200 * (obj_class == 0))
            draw.rectangle([x_min, y_min, x_max, y_max], outline=color, width=3)

            # Prepare the text annotation: confidence and class
            annotation_text = f'Class: {obj_class}, Conf: {conf:.2f}'

            # Calculate the size of the text box using `textbbox`
            if font:
                text_bbox = draw.textbbox((0, 0), annotation_text, font=font)
            else:
                text_bbox = draw.textbbox((0, 0), annotation_text)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]

            # Define the background rectangle for text
            text_background = (x_min, y_min - text_height - 2, x_min + text_width, y_min)

            # Draw the background rectangle for text
            draw.rectangle(text_background, fill=color)

            # Draw the annotation text on top of the box
            draw.text((x_min, y_min - text_height), annotation_text, fill='white', font=font)

    # Convert the annotated PIL image back to a Torch Tensor
    annotated_frame = torch.tensor(np.array(pil_frame), dtype=torch.uint8)
    # annotated_frame = annotated_frame.permute(1, 2, 0)
    
    return annotated_frame




def write_info_in_frame_partial_contrast(frame, data_partial, pos_x, pos_y, name, rotate):
    opt_distr = np.array([0.3, 0.6, 0.07, 0.015, 0.015])
    pos_y_ = pos_y
    # for idx, data in enumerate(data_partial):
    #     data = data - opt_distr[idx]
    #     font = cv2.FONT_HERSHEY_SIMPLEX
    #     font_scale = 0.4
    #     color = (0,)  # White color for grayscale (single channel)
    #     thickness = 1
    #     text = f'{name} {idx}: {data:.3f}'
    #     (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    #     canvas = np.zeros((text_height + baseline, text_width), dtype=np.uint8)+255
    #     cv2.putText(canvas, text, (0, text_height), font, font_scale, color, thickness)
    #     canvas_rotated = canvas
    #     if rotate:
    #         canvas_rotated = cv2.rotate(canvas, cv2.ROTATE_180)
    #         canvas_rotated = cv2.flip(canvas_rotated, 1)
    #     height, width = frame.shape[:2]

    #     # Define the position for the text (top right corner)
    #     x_pos = width - text_width - 10  # Slight padding from the edge
    #     y_pos = pos_y_

    #     # Place the rotated text on the original image
    #     y1, y2 = y_pos, y_pos + canvas_rotated.shape[0]
    #     x1, x2 = x_pos, x_pos + canvas_rotated.shape[1]

    #     # Ensure the text fits within the image dimensions
    #     if y2 <= height and x2 <= width:
    #         frame[int(y1):int(y2), int(x1):int(x2), 0] = canvas_rotated
    #         frame[int(y1):int(y2), int(x1):int(x2), 1] = canvas_rotated
    #         frame[int(y1):int(y2), int(x1):int(x2), 2] = canvas_rotated
    #     pos_y_ -= 30

    dist = np.sum(np.square((data_partial-opt_distr)))
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    color = (0,)  # White color for grayscale (single channel)
    thickness = 1
    text = f'{name}: {dist:.3f}'
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    canvas = np.zeros((text_height + baseline, text_width), dtype=np.uint8)+255
    cv2.putText(canvas, text, (0, text_height), font, font_scale, color, thickness)
    canvas_rotated = canvas
    if rotate:
        canvas_rotated = cv2.rotate(canvas, cv2.ROTATE_180)
        canvas_rotated = cv2.flip(canvas_rotated, 1)    
    height, width = frame.shape[:2]

    # Define the position for the text (top right corner)
    x_pos = width - text_width - pos_x  # Slight padding from the edge
    y_pos = pos_y

    # Place the rotated text on the original image
    y1, y2 = y_pos, y_pos + canvas_rotated.shape[0]
    x1, x2 = x_pos, x_pos + canvas_rotated.shape[1]

    # Ensure the text fits within the image dimensions
    if y2 <= height and x2 <= width:
        frame[int(y1):int(y2), int(x1):int(x2), 0] = canvas_rotated
        frame[int(y1):int(y2), int(x1):int(x2), 1] = canvas_rotated
        frame[int(y1):int(y2), int(x1):int(x2), 2] = canvas_rotated

    return frame



def write_info_in_frame(frame, data, pos_x, pos_y, name, rotate):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    color = (0,)  # White color for grayscale (single channel)
    thickness = 1
    text = f'{name}: {data:.3f}'
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    canvas = np.zeros((text_height + baseline, text_width), dtype=np.uint8)+255
    cv2.putText(canvas, text, (0, text_height), font, font_scale, color, thickness)
    canvas_rotated = canvas
    if rotate:
        canvas_rotated = cv2.rotate(canvas, cv2.ROTATE_180)
        canvas_rotated = cv2.flip(canvas_rotated, 1)    
    height, width = frame.shape[:2]

    # Define the position for the text (top right corner)
    x_pos = width - text_width - pos_x  # Slight padding from the edge
    y_pos = pos_y

    # Place the rotated text on the original image
    y1, y2 = y_pos, y_pos + canvas_rotated.shape[0]
    x1, x2 = x_pos, x_pos + canvas_rotated.shape[1]

    # Ensure the text fits within the image dimensions
    if y2 <= height and x2 <= width:
        frame[int(y1):int(y2), int(x1):int(x2), 0] = canvas_rotated
        frame[int(y1):int(y2), int(x1):int(x2), 1] = canvas_rotated
        frame[int(y1):int(y2), int(x1):int(x2), 2] = canvas_rotated

    return frame




def write_in_frame(frame, plot_info, data, idx, rotate):
    height, width = frame.shape[:2]
    if plot_info != None and plot_info != 'all':
        number_of_spaces = len(data) + 1
        for id, data_key in enumerate(data.keys()):
            if data_key == 'partial_contrast':
                frame = write_info_in_frame_partial_contrast(frame, data[data_key][idx], (1+id)*height/number_of_spaces, 10, data_key, rotate)
            else:
                frame = write_info_in_frame(frame, data[data_key][idx], (1+id)*height/number_of_spaces, 10, data_key, rotate)

    if plot_info == ['all']:
        d_entropy = data[0][idx]
        d_partial_contr = data[1][idx]
        d_std_C = data[2][idx]
        frame = write_info_in_frame(frame, d_entropy, height/4, 10, 'entr', rotate)
        frame = write_info_in_frame_partial_contrast(frame, d_partial_contr, 2*height/4, 10, 'part_C', rotate)
        frame = write_info_in_frame(frame, d_std_C, 3*height/4, 10, 'std_C', rotate)
    return frame






def plot_distr_distance(dataset, optimal_distr, vis_type='', image_type='frames'):
    optim_distr_mean, optim_distr_std = optimal_distr
    vis_type = '/' + vis_type 
    path = f'../info_{dataset}/info_data/partial_contrast/{image_type}{vis_type}'
    files = os.listdir(path)
    for file in files:
        if 'inter' not in file or '00_b' not in file:
            continue
        print(file)
        save_name = file.split('.')[0]
        distr = np.load(path+file)
        distance = np.sum(np.square((distr-optim_distr_mean)/(1)), axis=1)
        colors = ['red' if d > 0.025 else 'blue' for d in distance]
        for i, c in enumerate(colors):
            if c == 'red':
                print(i)

        plt.figure()
        plt.scatter(np.linspace(900, 900+len(distance[900:1000])-1, len(distance[900:1000])), distance[900:1000], marker='.', s=6, c=colors[900:1000])
        plt.title(f'Distribution distance from optimal')
        plt.xlabel('Time')
        plt.ylabel('Distance')
        plt.savefig(f'../info_{dataset}/temp_files/{save_name}-d15')
        plt.close()
