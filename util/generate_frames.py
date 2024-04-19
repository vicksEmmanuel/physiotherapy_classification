import os
import cv2
import csv
import json

from actions import Action
import shutil


def normalize_coordinates(corner, width, height):
    x, y = corner
    normalized_x = x / width
    normalized_y = y / height
    return (normalized_x, normalized_y)

# This function generates the csv data that holds the frames, frames location, and video properties
def generate_frames(data_path, dataset_type):

    save_path = f"data/frames_dataset/frame_lists/{dataset_type}.csv"
    annotation_save_path = f"data/frames_dataset/annotations/{dataset_type}.csv"
    # Create a new CSV file with a clean state
    csv_file = open(save_path, 'w', newline='')
    csv_writer = csv.writer(csv_file, delimiter=' ')
    csv_writer.writerow(['original_vido_id', 'video_id', 'frame_id', 'path', 'labels'])
    csv_file.close()

    # Loop through the folders in the video folder
    for folder_name in os.listdir(data_path):
        json_file_path = os.path.join(data_path, folder_name)
        with open(json_file_path) as json_file:
            data = json.load(json_file)
        # Extract the videoId and framesCount from the JSON data
        video_id = data['videoId']
        frames_count = data['framesCount']
        video_name = data['videoName']
        image_width = data['size']['width']
        image_height = data['size']['height']

        objects = data['objects'] # contains a list of objects in the video such as id, classTitle
        

        for frame in data['frames']:
            frame_index = frame['index']
            figures = frame['figures']
            
            # Process the figures in the frame
            for figure in figures:
                figure_id = figure['id']
                class_id = figure['classId']
                object_id = figure['objectId']
                description = figure['description']
                geometry_type = figure['geometryType']
                labeler_login = figure['labelerLogin']
                created_at = figure['createdAt']
                updated_at = figure['updatedAt']
                geometry = figure['geometry']
                
                # Process the geometry points
                exterior_points = geometry['points']['exterior']
                interior_points = geometry['points']['interior']

                class_title = None

                # Find object_id in objects
                for obj in objects:
                    if obj['id'] == object_id:
                        class_title = obj['classTitle']
                        break

                class_id = None

                for index in range(0, len(Action().action)):
                    if Action().action[index] == class_title:
                        class_id = index
                        break
                
                csv_file = open(annotation_save_path, 'a', newline='')
                csv_writer = csv.writer(csv_file)

                padding = len(str(frames_count))
                frame_number = str(frame_index).zfill(padding)


                width_scale = 256 / image_width
                height_scale = 256 / image_height


              

                # Function to transform a bounding box
                # def transform_bounding_box(box, width_scale, height_scale):
                #     (x1, y1), (x2, y2) = box['exterior']
                #     new_x1 = x1 * width_scale
                #     new_y1 = y1 * height_scale
                #     new_x2 = x2 * width_scale
                #     new_y2 = y2 * height_scale
                #     return [(new_x1, new_y1), (new_x2, new_y2)]
                

                # figure['geometry']['points']['exterior'] = transform_bounding_box(
                #     figure['geometry']['points'],
                #     width_scale,
                #     height_scale
                # )

                # x1, y1 = figure['geometry']['points']['exterior'][0]
                # x2, y2 = figure['geometry']['points']['exterior'][1]


                # normalized_top_left = normalize_coordinates((x1, y1), 256, 256)
                # normalized_bottom_right = normalize_coordinates((x2, y2), 256, 256)

                # print(f"Normalized top left: {normalized_top_left} Normalized bottom right: {normalized_bottom_right}")
                
                # x1, y1 = normalized_top_left
                # x2, y2 = normalized_bottom_right



                (x1, y1), (x2, y2) = figure['geometry']['points']['exterior']
                
                # Calculate the center of the bounding box
                x_center = (x1 + x2) / 2.0
                y_center = (y1 + y2) / 2.0
                
                # Calculate the width and height of the bounding box
                width = x2 - x1
                height = y2 - y1
                
                # Normalize the center coordinates and dimensions
                normalized_x_center = x_center / image_width
                normalized_y_center = y_center / image_height
                normalized_width = width / image_width
                normalized_height = height / image_height

                x1 = float(f"{normalized_x_center:.4f}")
                y1 = float(f"{normalized_y_center:.4f}")
                x2 = float(f"{normalized_width:.4f}")
                y2 = float(f"{normalized_height:.4f}")


                print(f"{x1} {y1} {x2} {y2} =====>")


                csv_writer.writerow([
                    video_id, frame_number,
                    x1, y1, x2, y2,
                    class_id,
                    65
                ])

                csv_file.close()


            csv_file = open(save_path, 'a', newline='')
            csv_writer = csv.writer(csv_file, delimiter=' ')
            padding = len(str(frames_count))
            frame_number = str(frame_index).zfill(padding)
            file_name  = f"data/frames/{video_id}/{video_id}_00{frame_number}.jpg"
            csv_writer.writerow([video_id, video_id, frame_index, file_name, "''"])
            csv_file.close()


def clear_folders(path):
    # Clear the frames dataset folder
    frames_dataset_folder = path
    if os.path.exists(frames_dataset_folder):
        shutil.rmtree(frames_dataset_folder)
    os.makedirs(frames_dataset_folder)

def process_data():
    # clear_folders("data/frames_dataset/annotations")
    clear_folders("data/frames_dataset/frame_lists")

    generate_frames("data/video_actions_annotated_data/train", 'train')
    generate_frames("data/video_actions_annotated_data/test", 'test')
    generate_frames("data/video_actions_annotated_data/val", 'val')


process_data()