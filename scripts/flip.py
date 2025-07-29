import cv2
import os
import glob

img_dir = r"E:\InternshipProject\images\viparita karani"

images = glob.glob(os.path.join(img_dir, "*.jpg"))

for img_path in images:
    img = cv2.imread(img_path)
    if img is None:
        continue

    flipped_img = cv2.flip(img, 1)

    base = os.path.basename(img_path)
    name, ext = os.path.splitext(base)
    flipped_name = f"{name}_flipped{ext}"
    img_dir = os.path.dirname(img_path)
    flipped_path = os.path.join(img_dir, flipped_name)

    cv2.imwrite(flipped_path, flipped_img)
print("Successful")