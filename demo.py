import cv2
import numpy as np
import torch
from torchvision import transforms
from model import VideoClassifier
import argparse

def process_video(input_file, output_file, frame_jump, num_classes, lstm_hidden_size, lstm_num_layers, model_path):
    model = VideoClassifier(num_classes=num_classes, lstm_hidden_size=lstm_hidden_size, lstm_num_layers=lstm_num_layers)
    model.load_state_dict(torch.load(model_path))
    model = model.cuda()
    model.eval()

    cap = cv2.VideoCapture(input_file)
    if not cap.isOpened():
        print("Error opening video stream or file")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    transform = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    probabilities = []
    frame_indices = []

    while True:
        frames = []
        for _ in range(45):
            ret, frame = cap.read()
            if not ret:
                break
            frame_indices.append(int(cap.get(cv2.CAP_PROP_POS_FRAMES)))
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_tensor = transform(frame_rgb).unsqueeze(0)
            frames.append(frame_tensor)

        if len(frames) == 45:
            frames = torch.cat(frames).unsqueeze(0).cuda()
            with torch.no_grad():
                outputs = model(frames)
                probabilities.append(torch.softmax(outputs, dim=1).cpu().numpy())

        cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_POS_FRAMES) - 45 + frame_jump)

        if not ret:
            break

    cap.release()

    probabilities = np.concatenate(probabilities)
    avg_probabilities = np.mean(probabilities, axis=0)

    # Draw probabilities on video
    cap = cv2.VideoCapture(input_file)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        if frame_idx in frame_indices:
            idx = frame_indices.index(frame_idx)
            prob = avg_probabilities[idx]

            label = f"Probabilities: {prob[0]:.2f}, {prob[1]:.2f}"
            cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        out.write(frame)

    cap.release()
    out.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process a video using a trained model.')
    parser.add_argument('--input_file', type=str, required=True, help='Path to the input video file')
    parser.add_argument('--output_file', type=str, required=True, help='Path to the output video file')
    parser.add_argument('--frame_jump', type=int, default=20, help='Number of frames to jump after processing a set of frames')
    parser.add_argument('--num_classes', type=int, default=2, help='Number of output classes')
    parser.add_argument('--lstm_hidden_size', type=int, default=512, help='Hidden size of LSTM in the model')
    parser.add_argument('--lstm_num_layers', type=int, default=2, help='Number of LSTM layers in the model')
    parser.add_argument('--model_path', type=str, default='best_model.pth', help='Path to the trained model file')

    args = parser.parse_args()
    process_video(args.input_file, args.output_file, args.frame_jump, args.num_classes, args.lstm_hidden_size, args.lstm_num_layers, args.model_path)
