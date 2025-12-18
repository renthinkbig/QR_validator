import streamlit as st
import numpy as np
import cv2
from PIL import Image
from transformers import pipeline
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

@st.cache_resource
def load_depthanything_model():
    pipe = pipeline(
        task="depth-estimation",
        model="depth-anything/Depth-Anything-V2-Small-hf",   # your local directory
    )
    return pipe
# pipe = pipeline(task="depth-estimation", model="./depthanything-v2")

class QRProcessor(VideoProcessorBase):
    detected_frames = 0
    captured = False
    qr_crop = None

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        if self.captured:
            # Show frozen QR region only
            return frame.from_ndarray(self.qr_crop, format="bgr24")

        data, bbox, _ = qr_detector.detectAndDecode(img)

        if bbox is not None:
            pts = bbox.astype(int).reshape(-1, 2)

            # Draw box
            for i in range(4):
                cv2.line(img, tuple(pts[i]), tuple(pts[(i+1) % 4]), (0, 255, 0), 2)

            self.detected_frames += 1

            if self.detected_frames > 8:
                # Crop QR region
                x_min = pts[:, 0].min()
                y_min = pts[:, 1].min()
                x_max = pts[:, 0].max()
                y_max = pts[:, 1].max()

                self.qr_crop = img[y_min:y_max, x_min:x_max]
                self.captured = True

                return frame.from_ndarray(self.qr_crop, format="bgr24")
        else:
            self.detected_frames = 0

        return frame.from_ndarray(img, format="bgr24")

def depth_estimation(image):
    depth = pipe(image)["depth"]
    # cv2.imwrite('depth.png',np.array(depth))
    W_real = 5.1  # Real width of cardboard in cm (example)
    depth_float = np.array(depth).astype("float32")
    l, b = image.size
    # w is pixel width of cardboard region
    W_px = l  # width in pixels
    # ------------------------
    # Step 2: Compute scale factor
    # ------------------------
    scale = W_real / W_px  # cm per pixel
    print("Scale factor:", scale, "cm per depth unit")

    # ------------------------
    # Step 3: Compute actual depth map
    # ------------------------
    depth_metric = depth_float * scale

    return depth_metric

def is_single_depth(depth_img, tol=2):
    # Flatten the array
    d = depth_img.flatten()

    # Compare all values against the first pixel within tolerance
    return np.all(np.abs(d - d[0]) < tol)

st.set_page_config(page_title="QR Capture", layout="centered")

st.title("📸 QR Capture Page")
st.write("Point your camera at the QR code. Then click **Capture**.")

# Streamlit camera input
#captured_image = st.camera_input("Camera")
#captured_image=st.file_uploader('upload image')
webrtc_streamer(
    key="qr-camera",
    video_processor_factory=QRProcessor,
    media_stream_constraints={"video": True, "audio": False},
)
pipe= load_depthanything_model()
if QRProcessor.captured and QRProcessor.qr_crop is not None:
    qr_img = QRProcessor.qr_crop

    st.subheader("Captured QR Region")
    st.image(qr_img, channels="BGR")

    # Save locally
    #cv2.imwrite("captured_qr.png", cv2.cvtColor(qr_img, cv2.COLOR_RGB2BGR))
    #st.success("Image saved locally as captured_qr.png")

    if st.button("Upload"):
            # st.image(qr_img)
            h, w = qr_img.shape[:2]

            # Crop central region (say 60-70% of QR code)
            center_ratio = 0.7
            cx, cy = w // 2, h // 2
            half_w, half_h = int(w * center_ratio / 2), int(h * center_ratio / 2)

            central_region = qr_img[cy - half_h:cy + half_h, cx - half_w:cx + half_w]

            # Convert to HSV for better color segmentation
            hsv = cv2.cvtColor(central_region, cv2.COLOR_RGB2HSV)

            # Compute mask for areas that differ from QR code color (assume QR code is mostly black/white)
            # Here we assume the inserted image has non-white/non-black background
            lower = np.array([0, 30, 30])  # adjust thresholds if needed
            upper = np.array([180, 255, 220])
            mask = cv2.inRange(hsv, lower, upper)

            # Find contours in the masked area
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                # Assume largest contour is the inserted image
                max_contour = max(contours, key=cv2.contourArea)
                x, y, w_box, h_box = cv2.boundingRect(max_contour)

                # Crop inserted image
                insert_area = central_region[y:y + h_box, x:x + w_box]
                roi=Image.fromarray(insert_area)
                st.image(roi)
                # depth_metric=depth_estimation(roi)
                depth = pipe(roi)["depth"]
                # st.image(np.array(depth))
                laplacian = cv2.Laplacian(np.array(depth), cv2.CV_64F)
                laplacian_display = np.uint8(np.absolute(laplacian))
                # st.image(laplacian_display)
                score = laplacian.var()
                st.write(score)
                if score < 330:
                    st.write('No depth- Fake QR')
                else:
                    st.write('Depth detected- Valid QR')

            else:
                st.error('No ROI detected')

#else:
           # st.error('Failed to detect QR region!!! Recapture the image.')







