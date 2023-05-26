# Sauron-IX-III
Open Source CCTV A.I Project

About Voiture Tampon Multiple Cameras file :
Python Script to detect cars and how long they stay parked, display a windows to show the result live, the gradient color formula has to be improved.
The Camera URL is Axis's camera feel free to adapt it to your needs.
The camera configuration file has to be in a subfolder called conf and the structure has to be like this :
[
  {
    "Camera": "Name of the camera",
    "IP": "IP Address",
    "Quad": 0,
    "Localisation": [0,0],
    "Login": "your camera login",
    "Password": "your camera password"
  }
]
