import cv2
import time
from tqdm import tqdm
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector
from datetime import datetime
import ffmpeg
import os

class Movie_Scene_fetcher:
    
    movies_set = None
    full_movie_path = None
    
    def __init__(self, movie_path_1, movie_path_2):
        print('Loading Movies...')
        self.movies_set_1 = self.get_scene_frame(movie_path_1)
        self.movies_set_2 = self.get_scene_frame(movie_path_2)
        self.full_movie_path_1 = movie_path_1
        self.full_movie_path_2 = movie_path_2
        print("Movies Loaded Successfully")
        
    def timecode_to_seconds(self, timecode):
        td = datetime.strptime(timecode, '%H:%M:%S.%f') - datetime(1900, 1, 1)
        seconds = td.total_seconds()
        return seconds
    
    def get_scene_frame(self, scene_path):
        video_manager = VideoManager([scene_path])
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector())

        # Start video manager and detect scenes
        video_manager.start()
        scene_manager.detect_scenes(frame_source=video_manager)

        scene_list = scene_manager.get_scene_list()

        cap = cv2.VideoCapture(scene_path)
        frame_list = []
        for scene in tqdm(scene_list):
            
            initial_time = self.timecode_to_seconds(scene[0].get_timecode())
            final_time = self.timecode_to_seconds(scene[1].get_timecode())
            time_between = final_time - initial_time
            incre_time = time_between/2
            frame_time = initial_time + incre_time
            cap.set(cv2.CAP_PROP_POS_MSEC, frame_time * 1000)
            ret, frame = cap.read()
            if not ret: print("Error reading frame")
            frame_list.append((frame,frame_time,time_between))        
            
        cap.release()
        cv2.destroyAllWindows()
        return frame_list
    
    def compressFrame(self,frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        width = int(frame.shape[1] * 0.5)
        height = int(frame.shape[0] * 0.5)
        frame = cv2.resize(frame, (width, height))
        frame = cv2.GaussianBlur(frame, (5, 5), 0)
        return frame

    def fetch_scene_from_movies(self):
        frame_details = []
        
        # Compare scenes from both movie sets
        for movie_set, movie_path, movie_index in [(self.movies_set_1, self.full_movie_path_1, 1), 
                                                   (self.movies_set_2, self.full_movie_path_2, 2)]:
            for frame_scene, frame_scene_time, fram_scene_duration in movie_set:
                compressed_frame_scene = self.compressFrame(frame_scene)
                
                highest_similarity = 0
                scene_detail = []
                movie_detail = []
                for other_frame_scene, other_frame_scene_time, other_frame_scene_duration in self.movies_set_1 if movie_index == 2 else self.movies_set_2:
                    compressed_other_frame_scene = self.compressFrame(other_frame_scene)
                    result = cv2.matchTemplate(compressed_frame_scene, compressed_other_frame_scene, cv2.TM_CCOEFF_NORMED)
                    max_val = cv2.minMaxLoc(result)[1]
                    similarity_score = max_val * 100
                    
                    if similarity_score > highest_similarity:
                        highest_similarity = similarity_score
                        scene_detail = [frame_scene, frame_scene_time, fram_scene_duration]
                        movie_detail = [other_frame_scene, other_frame_scene_time, other_frame_scene_duration]
                
                frame_details.append([highest_similarity, scene_detail, movie_detail])
         
        return frame_details

    def split_clip(self, video_path, start_time, duration, output_path):
        ffmpeg.input(video_path, ss=start_time, t=duration).output(output_path).run(overwrite_output=True)

    def merge_clips(self, clip_paths, output_path):
        inputs = [ffmpeg.input(clip).video for clip in clip_paths]
        merged = ffmpeg.concat(*inputs, v=1, a=0)
        merged.output(output_path).run(overwrite_output=True)
        
    def create_clone(self, frame_details):
        clip_paths = []
        for frame_detail in frame_details:
            highest_similarity = frame_detail[0]
            frame_scene, frame_scene_time, fram_scene_duration = frame_detail[1]
            frame_movie_scene, frame_movie_scene_time, frame_movie_scene_duration = frame_detail[2]
            
            os.makedirs("Trial_clips", exist_ok=True)
            output_clip_path = os.path.join("Trial_clips", "clip_{}.mp4".format(frame_movie_scene_time))
            initial_time = frame_movie_scene_time - frame_movie_scene_duration/2
            self.split_clip(self.full_movie_path_1 if highest_similarity == frame_detail[0] else self.full_movie_path_2, initial_time, frame_movie_scene_duration, output_clip_path)
            clip_paths.append(output_clip_path)
        
        self.merge_clips(clip_paths, 'merged_output.mp4')
        print("Clips Merged Successfully")
            
# Usage
video_path_1 = 'path_to_video_1.mp4'
video_path_2 = 'path_to_video_2.mp4'

movie_scene_fetcher = Movie_Scene_fetcher(video_path_1, video_path_2)
frame_details = movie_scene_fetcher.fetch_scene_from_movies()
movie_scene_fetcher.create_clone(frame_details)
