import os
from flask import Flask, render_template, request, redirect, url_for
import cv2
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'origin_image' not in request.files or 'concat_image' not in request.files:
        return redirect(request.url)
    
    origin_image = request.files['origin_image']
    concat_image = request.files['concat_image']
    
    if origin_image.filename == '' or concat_image.filename == '':
        return redirect(request.url)
    
    origin_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(origin_image.filename))
    concat_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(concat_image.filename))
    
    # Ensure the upload directory exists
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    origin_image.save(origin_path)
    concat_image.save(concat_path)
    
    try:
        # Process the images using the feature detection code
        result_path = process_images(origin_path, concat_path)
    except ValueError as e:
        return render_template('error.html', error_message=str(e))
    
    return render_template('result.html', result_image=result_path)

def process_images(origin_path, concat_path):
    img1 = cv2.imread(origin_path, 0)
    img2 = cv2.imread(concat_path, 0)

    if img1 is None:
        raise FileNotFoundError(f"Image at '{origin_path}' not found or unable to load.")
    if img2 is None:
        raise FileNotFoundError(f"Image at '{concat_path}' not found or unable to load.")

    orb = cv2.ORB_create(nfeatures=1000)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    if des1 is None or des2 is None:
        raise ValueError("One of the images does not have enough features to match.")

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])

    img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)
    result_path = os.path.join(app.config['UPLOAD_FOLDER'], 'result.jpg')
    cv2.imwrite(result_path, img3)
    
    # Return the path relative to the static directory
    return 'uploads/result.jpg'

if __name__ == '__main__':
    app.run(debug=True)