from utils import CentroidTracker, VideoBufferThread

class Camera:
    """Camera class to handle multiple cameras stream.
    """
    def __init__(self, dict_cam, config, webhook_teams, time_dispappeared = 10, similarity_threshold = 0.5) :
        """Initialize the Camera class.

        Args:
            dict_cam (dict): The dictionnary containing the camera information.
            config (dict): The dictionnary containing the configuration information.
            webhook_teams (str): The webhook url to send the alert.
            time_dispappeared (int, optional): The time in seconds before an object is considered disappeared. Defaults to 10.
            similarity_threshold (float, optional): The similarity threshold to consider two objects as the same. Defaults to 0.5.
        """
        self.name = dict_cam["Camera"]
        self.ip = dict_cam["IP"]
        self.quad = dict_cam["Quad"]
        self.login = dict_cam["Login"]
        self.password = dict_cam["Password"]
        self.streams: list[VideoBufferThread] = []
        self.centroid_tracker: list[CentroidTracker] = []
        self.config = config
        self.start(webhook_teams, time_dispappeared, similarity_threshold)

    def start(self, webhook_teams, time_dispappeared, similarity_threshold):
        """Add a camera to the camera list.
        """
        nb_cam = 5 if self.quad else 2
        for i in range(1, nb_cam ):
            url = self.build_url(i)
            video_buffer = VideoBufferThread(url)
            video_buffer.start()
            self.streams.append(video_buffer)
            camera_name = f"{self.name}_{i}" if self.quad else self.name
            access_key = self.config["aws_access_key"]
            secret_key = self.config["aws_secret_key"]
            bucket_name = self.config["aws_bucket_name"]
            self.centroid_tracker.append(CentroidTracker(camera_name, (access_key, secret_key, bucket_name, webhook_teams), time_dispappeared, similarity_threshold))
        
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
            stream.wait_first_frame()

    def get_frame(self, cam_indice = 0) :
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
        
        return self.streams[cam_indice].get_latest_frame(), self.centroid_tracker[cam_indice]
    
    def stop(self):
        for stream in self.streams :
            stream.stop()
            stream.join()