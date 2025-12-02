import streamlit as st
import numpy as np
import cv2
from PIL import Image
from transformers import pipeline

@st.cache_resource
def load_depthanything_model():
    pipe = pipeline(
        task="depth-estimation",
        model="depth-anything/Depth-Anything-V2-Small-hf",   # your local directory
    )
    return pipe
# pipe = pipeline(task="depth-estimation", model="./depthanything-v2")

def depth_estimation(pipe,image):
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
captured_image = st.camera_input("Camera")
#captured_image=st.file_uploader('upload image')
pipe= load_depthanything_model()
if captured_image is not None:
    # Convert to OpenCV format
    img = Image.open(captured_image)
    img = np.array(img)

    st.subheader("Captured Image:")
    st.image(img, channels="RGB")

    # Save locally
    # cv2.imwrite("captured_qr.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    # st.success("Image saved locally as captured_qr.png")

    if st.button("Upload"):
        detector = cv2.QRCodeDetector()
        data, points, _ = detector.detectAndDecode(img)

        if points is not None:
            # Get bounding box of QR code
            points = points[0].astype(int)
            x_min = points[:, 0].min()
            y_min = points[:, 1].min()
            x_max = points[:, 0].max()
            y_max = points[:, 1].max()

            # Crop QR code region
            qr_crop = img[y_min:y_max, x_min:x_max]
            # st.image(qr_crop)
            h, w = img.shape[:2]

            # Crop central region (say 60-70% of QR code)
            center_ratio = 0.7
            cx, cy = w // 2, h // 2
            half_w, half_h = int(w * center_ratio / 2), int(h * center_ratio / 2)

            central_region = img[cy - half_h:cy + half_h, cx - half_w:cx + half_w]

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
                depth_metric=depth_estimation(pipe,roi)

                check = is_single_depth(depth_metric)
                if check:
                    st.write('No depth- Fake QR')
                else:
                    st.write('Depth detected- Valid QR')
            else:
                st.error('No ROI detected')

        else:
            st.error('Failed to detect QR region!!! Recapture the image.')


