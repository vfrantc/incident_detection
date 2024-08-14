import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from model import VideoClassifier
from PIL import Image
from argparse import ArgumentParser

def custom_mapping(x):
    if x <= 0.79:
        return 0
    elif x >= 0.8:
        return 1
    else:
        t = (x - 0.79) / 0.01
        return 3 * t**2 - 2 * t**3

def draw_dashed_line(img, pt1, pt2, color, thickness=1, gap=10):
    dist = ((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) ** .5
    pts = np.linspace(pt1, pt2, round(dist / gap), dtype=int)
    for i in range(0, len(pts), 2):
        if i + 1 < len(pts):
            cv2.line(img, tuple(pts[i]), tuple(pts[i + 1]), color, thickness)

def draw_graph(img, values, current_frame, total_frames, v_position=0.8, graph_height=0.2, label="Estimated probability of accident"):
    img_height, img_width, _ = img.shape
    graph_top = int(img_height * (1 - v_position))
    graph_bottom = int(graph_top + img_height * graph_height)
    graph_height = graph_bottom - graph_top
    points = []

    for index, value in enumerate(values):
        x = int(img_width * (index / total_frames))
        y = int(graph_bottom - graph_height * value)
        points.append((x, y))

    for i in range(1, len(points)):
        color = (0, 0, 255) if values[i] > 0.95 else (0, 255, 0)
        cv2.line(img, points[i - 1], points[i], color, thickness=2)

    draw_dashed_line(img, (0, graph_bottom), (img_width, graph_bottom), (0, 255, 255), thickness=2)
    draw_dashed_line(img, (0, graph_top), (img_width, graph_top), (0, 255, 255), thickness=2)
    current_pos = int(img_width * (current_frame / total_frames))
    cv2.line(img, (current_pos, graph_top), (current_pos, graph_bottom), (0, 0, 255), thickness=2)
    textsize = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
    textX = (img.shape[1] - textsize[0]) // 2
    cv2.putText(img, label, (textX, graph_top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(img, '1', (10, graph_top - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(img, '0', (10, graph_bottom - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    return img

def process_video(input_file, output_file, frame_jump=20):
    cap = cv2.VideoCapture(input_file)
    if not cap.isOpened():
        print("Error opening video stream or file")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'mp4v'), fps, (1280, 720))

    transform = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    probabilities = []
    frame_indices = []
    sliding_window = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_indices.append(int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)
        frame_tensor = transform(frame_pil).unsqueeze(0)
        sliding_window.append(frame_tensor)

        if len(sliding_window) > 45:
            sliding_window.pop(0)

        if len(sliding_window) == 45:
            frames = torch.cat(sliding_window).unsqueeze(0).cuda()
            with torch.no_grad():
                #outputs1 = model1(frames)
                outputs2 = model2(frames)
                #avg_prob = (torch.softmax(outputs1, dim=1)[:, 1] + torch.softmax(outputs2, dim=1)[:, 1]) / 2
                avg_prop = torch.softmax(outputs2, dim=1)[:, 1]
                probabilities.append(avg_prob.cpu().numpy())

    cap.release()

    probabilities = np.concatenate(probabilities)
    probabilities = np.pad(probabilities, (0, num_frames - len(probabilities)), mode='constant')

    #mapped_probabilities = [custom_mapping(x) for x in probabilities]
    mapped_probabilities = [x for x in probabilities]
    cap = cv2.VideoCapture(input_file)
    prob_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        frame_resized = cv2.resize(frame, (1280, 720))

        if frame_idx in frame_indices and prob_idx < len(mapped_probabilities):
            prob = mapped_probabilities[prob_idx]
            label = f"Probability: {prob:.2f}"
            cv2.putText(frame_resized, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            prob_idx += 1

        frame_with_graph = draw_graph(frame_resized, mapped_probabilities[:frame_idx + 1], frame_idx, num_frames, v_position=0.8, graph_height=0.2)
        out.write(frame_with_graph)

    cap.release()
    out.release()

def main(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(".mp4") or filename.endswith(".avi"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            process_video(input_path, output_path)

if __name__ == "__main__":
    parser = ArgumentParser(description="Process videos with a pretrained model and add probability graphs.")
    parser.add_argument("--input_folder", type=str, required=True, help="Path to the input folder containing videos.")
    parser.add_argument("--output_folder", type=str, required=True, help="Path to the output folder for processed videos.")
    args = parser.parse_args()

    # Load the models
    model1 = VideoClassifier(num_classes=2, lstm_hidden_size=512, lstm_num_layers=2)
    model1.load_state_dict(torch.load('weights/best_model_002_320x320_1.pth'))
    model1 = model1.cuda()
    model1.eval()

    model2 = VideoClassifier(num_classes=2, lstm_hidden_size=512, lstm_num_layers=2)
    model2.load_state_dict(torch.load('weights/best_model_002_320x320_2.pth'))
    model2 = model2.cuda()
    model2.eval()

    main(args.input_folder, args.output_folder)
