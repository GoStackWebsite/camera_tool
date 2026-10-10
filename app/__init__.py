"""People-activity analytics for an RTSP camera."""
import os

# Must be set before cv2 is imported anywhere: force RTSP over TCP (no packet loss
# artefacts) and a 5s socket timeout so a dead camera is detected instead of hanging.
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp|timeout;5000000")
# Only show fatal FFmpeg messages (hides harmless HEVC "POC"/"PPS" warnings at stream start).
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "8")
