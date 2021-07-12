from flask import Flask, render_template, request
from PIL import Image
import os
import cartoonize
app = Flask("__name__")
cartoonized_images_folder_path = os.path.join('static', 'cartoonized_images')
app.config['UPLOAD_FOLDER'] = cartoonized_images_folder_path

@app.route("/", methods=['POST', 'GET'])
def home():
    if request.method == "POST":
        image_url = request.files["input_image_url"]
        output_image, process_time = cartoonize.web_cartoonization("hayao", image_url)
        cartoonized_image_path = os.path.join(app.config['UPLOAD_FOLDER'], str(image_url.filename))
        output_image.save(cartoonized_image_path)
        return render_template("index.html", user_image=cartoonized_image_path)
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)