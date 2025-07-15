from flask import Flask, render_template, jsonify
import os
import threading
import time
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
from flask import send_from_directory

# Define the path for the Excel file
excel_file_path = 'D:/ANAND Project/Haldex Break Chember TY24/UI Break Chember/Result Excel File/processing_results.xlsx'

# Create the Excel file with the necessary columns if it doesn't exist
if not os.path.exists(excel_file_path):
    df = pd.DataFrame(columns=['Serial No', 'Image name', 'EZE-PLUG', 'HEX-NUT', 'NUT-HEX','PIN-COTTER','Timestamp'])
    df.to_excel(excel_file_path, index=False)
    

# Load the pre-trained models
plug_model = load_model('eze_plug_model.h5')
hexnut_model = load_model('hex_nut_model.h5')
nuthex_model = load_model('nut_hex_model.h5')
pincotter_model = load_model('pin_cotter_model.h5')

#m4_model = load_model('m4_model.h5')

# Coordinates for each component
eze_plug_coordinates = {"x": 647.5, "y": 797.0000000000002, "width": 500.0, "height": 490.0}
hex_nut_coordinates = {"x": 962.5, "y": 3209.5, "width": 400.0, "height": 315.0}
nut_hex_coordinatesl = {"x": 523.0, "y": 2180.0, "width": 385.0, "height": 649.9999999999998}
nut_hex_coordinatesr = {"x": 1565.0, "y": 2207.0, "width": 385.0, "height": 649.9999999999998}
pin_cotter_coordinates = {"x": 875.0, "y": 3842.0, "width": 385.0, "height": 440.0}



app = Flask(__name__)

# Global variable to store the latest image path
dynamic_latest_image_path = ""
absolute_latest_image_path =""

###############################################################################################################################
#Home page Code 
# Define a route to serve the HTML file at the root URL
@app.route('/')
def index():
    return render_template('Trial.html')  # Ensure you have a 'Trial.html' file in a 'templates' folder

#################################################################################################################################

# Route to fetch the latest image from a specified folder
@app.route('/latest-image')
def latest_image():
    global dynamic_latest_image_path  # Declare the global variable
    global absolute_latest_image_path
    
    IMAGE_FOLDER = 'D:/ANAND Project/Haldex Break Chember TY24/UI Break Chember/static/images/'  # Update with your actual image folder path
    files = sorted(
        (f for f in os.listdir(IMAGE_FOLDER) if os.path.isfile(os.path.join(IMAGE_FOLDER, f))),
        key=lambda f: os.path.getmtime(os.path.join(IMAGE_FOLDER, f)),
        reverse=True
    )
    if files:
        dynamic_latest_image_path = f"/static/images/{files[0]}" #Dynamic path can not be used for the processing the image
        absolute_latest_image_path = os.path.join(IMAGE_FOLDER, files[0]) #Absolute path
        print("Latest Image Path updated in function:", dynamic_latest_image_path)  # Print the latest image path inside the function
        return jsonify({"imagePath": dynamic_latest_image_path})
    
    dynamic_latest_image_path = ""  # Set to empty if no images found
    print("No images found in the directory.")
    return jsonify({"imagePath": ""})

###############################################################################################################################

last_processed_image = ""


@app.route('/process-latest-image')
def process_latest_image():
    global absolute_latest_image_path, last_processed_image  # Access the global variable
            
     # Only proceed if there is a new image
    if absolute_latest_image_path and absolute_latest_image_path != last_processed_image:
        # Update the last processed image path
        last_processed_image = absolute_latest_image_path

        
        # Process the image using latest_image_path
        print("Processing image at path:", absolute_latest_image_path)
        
        output_folder = 'D:/ANAND Project/Haldex Break Chember TY24/UI Break Chember/Result/'
        img = cv2.imread(absolute_latest_image_path)
        image_name = os.path.basename(absolute_latest_image_path)
        
        # Prediction and annotation functions (from your code)
        def predict_and_draw(img, model, coordinates, label):
            x, y, width, height = int(coordinates['x']), int(coordinates['y']), int(coordinates['width']), int(coordinates['height'])
            roi = img[y:y + height, x:x + width]
            roi_resized = cv2.resize(roi, (224, 224)) / 255.0  # Normalize

            prediction = model.predict(np.expand_dims(roi_resized, axis=0))
            confidence = np.max(prediction) * 100
            predicted_class = np.argmax(prediction, axis=1)[0]
            display_text = f'{label}: {"Okay" if predicted_class == 1 else "Not Okay"} ({confidence:.2f}%)'

            x = x - 50
            
            height = height + 25
            width = width + 25
            
            color = (0, 255, 0) if predicted_class == 1 else (0, 0, 255)
            cv2.rectangle(img, (x, y), (x + width, y + height), color, 18)
            cv2.putText(img, display_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 4.0, color, 5)
            
            return predicted_class
        
        
               
        # Apply the models to the image and get results for each component
        plug_result = "Okay" if predict_and_draw(img, plug_model, eze_plug_coordinates, 'Plug') == 1 else "Not Okay"
        hexnut_result = "Okay" if predict_and_draw(img, hexnut_model, hex_nut_coordinates, 'Hex Nut') == 1 else "Not Okay"
        nuthexl_result = "Okay" if predict_and_draw(img, nuthex_model, nut_hex_coordinatesl, 'Nut Hex') == 1 else "Not Okay"
        #nuthexr_result = "Okay" if predict_and_draw(img, nuthex_model, nut_hex_coordinatesr, 'Nut Hex') == 1 else "Not Okay"
        pin_result = "Okay" if predict_and_draw(img, pincotter_model, pin_cotter_coordinates, 'Pin Cotter') == 1 else "Not Okay"
        
        #nuthex_result = "Okay" if nuthexl_result == "Okay" and nuthexr_result == "Okay" else "Not Okay"
                
        # Save the processed image with the same name as the original image
        output_path = os.path.join(output_folder, os.path.basename(absolute_latest_image_path))
        cv2.imwrite(output_path, img)
        
        # Load the existing Excel file and append the new row
        df = pd.read_excel(excel_file_path)
        new_row = {
            'Serial No': len(df) + 1,
            'Image name': image_name,
            'EZE-PLUG': plug_result,
            'HEX-NUT': hexnut_result,
            'NUT-HEX': nuthexl_result,
            'PIN-COOTER': pin_result,
            #'M4 Screw' : m4_result,
            'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Format as desired
        }
        df = df._append(new_row, ignore_index=True)
        
        # Save the updated DataFrame back to the Excel file
        df.to_excel(excel_file_path, index=False)
                           
        print("Processed image at path:", absolute_latest_image_path)
        # Add any processing logic here
        # Determine overall result based on individual component results

        overall_status = "Okay" if all(result == "Okay" for result in [plug_result, hexnut_result, nuthexl_result, pin_result]) else "Not Okay"
        #overall_status = "Okay" if all(result == "Okay" for result in [bush_result, nipple_result, oring_result, m4_result]) else "Not Okay"

        os.remove(absolute_latest_image_path)
        
        # Include the overall status in the response
        return jsonify({"message": f"Processed the latest image at path: {absolute_latest_image_path}", "overallStatus": overall_status})
    else:
        return "No latest image available to process."

#########################################################################################################################################
        
# Route to fetch the latest processed image from the result folder
@app.route('/latest-result-image')
def latest_result_image():
    result_folder = 'D:/ANAND Project/Haldex Break Chember TY24/UI Break Chember/Result/'  # Path to the result folder
    files = sorted(
        (f for f in os.listdir(result_folder) if os.path.isfile(os.path.join(result_folder, f))),
        key=lambda f: os.path.getmtime(os.path.join(result_folder, f)),
        reverse=True
    )
    if files:
        latest_result_image_path = f"/result/{files[0]}"
        print("Latest Result Image Path:", latest_result_image_path)
        return jsonify({"imagePath": latest_result_image_path})
    
    return jsonify({"imagePath": ""})

@app.route('/result/<path:filename>')
def serve_result_image(filename):
    result_directory = "D:/ANAND Project/Haldex Break Chember TY24/UI Break Chember/Result/"
    return send_from_directory(result_directory, filename)


if __name__ == '__main__':
    app.run(debug=True)
