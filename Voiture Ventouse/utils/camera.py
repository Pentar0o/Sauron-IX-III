from numpy import ndarray
from utils import CentroidTracker, VideoBufferThread

class Camera:
    """Camera class to handle multiple cameras stream.
    """
    def __init__(self, dict_cam, time_dispappeared, similarity_threshold) :
        """Initialize the Camera class.

        Args:
            dict_cam (dict): The dictionnary containing the camera information.
            time_dispappeared (int): The time in seconds before an object is considered disappeared. Defaults to 10.
        """
        self.name = dict_cam["Camera"]
        self.ip = dict_cam["IP"]
        self.quad = dict_cam["Quad"]
        self.login = dict_cam["Login"]
        self.password = dict_cam["Password"]
        self.streams: list[VideoBufferThread] = []
        self.centroid_tracker: list[CentroidTracker] = []
        self.start(time_dispappeared, similarity_threshold)

    def start(self, time_dispappeared: int, similarity_threshold: float):
        """Add a camera to the camera list.
        """
        nb_cam = 5 if self.quad else 2
        for i in range(1, nb_cam ):
            url = self.build_url(i)
            video_buffer = VideoBufferThread(url)
            video_buffer.start()
            self.streams.append(video_buffer)
            camera_name = f"{self.name}_{i}" if self.quad else self.name
            self.centroid_tracker.append(CentroidTracker(camera_name, time_dispappeared, similarity_threshold))
        
    def build_url(self, cam_indice = 0):
        """Build the URL of the camera.

        Returns:
            str: The URL of the camera.
        """
        base_url = f"rtsp://{self.login}:{self.password}@{self.ip}/axis-media/media.amp"
        if self.quad:
            return f"{base_url}?camera={cam_indice}"
        return base_url
    
    def wait_until_ready(self):
        """Wait until the camera is ready."""
        for stream in self.streams :
            started = stream.wait_first_frame()
            if not started : raise Exception("Error while connecting the RTSP stream")

    def get_frame(self, cam_indice = 0):
        """Get the latest frame from the camera.
            for the quad camera, the cam_indice is the camera number (1 to 4)
        
        Args:
            cam_indice (int, optional): The camera indice. Defaults to 0.
        
        Returns:
            (frame, centroid_tracker): The latest frame and the centroid tracker for the camera.
        """
        if cam_indice >= len(self.streams) :
            return None, None
        
        if not self.streams[cam_indice].is_alive() :
            return None, None
        
        return (self.streams[cam_indice].latest_frame, self.centroid_tracker[cam_indice])
    
    def stop(self):
        for stream in self.streams :
            stream.stop()
            stream.join()