import pickle
from flask import Flask, request, jsonify
import numpy as np

app = Flask(__name__)

# Load the pickled model
model = pickle.load(open('model.pkl', 'rb'))

# Define the brand_map and pixel_map
brand_map = {
    'Samsung             ': 0,
    'Sony                ': 1,
    'LG                  ': 2,
    'Toshiba             ': 3,
    'NA                  ': 4,
    'Hisense             ': 5,
    'Amazon fire TV      ': 6,
    'Insignia            ': 7,
    'TCL 75              ': 8,
    'VIZIO               ': 9,
    'Philips 75          ': 10,
    'TCL 6Series 65      ': 11,
    'RCA 32              ': 12,
    'JVC 58              ': 13
}

pixel_map = {
    '4K': 0,
    '1080p': 1,
    '720p': 2,
    '8K': 3,
    '4k': 4,
    '720P': 5,
    'Ultra HD': 6
}


@app.route('/predict', methods=['POST'])
def predict():
    # Get the JSON input data
    data = request.get_json()
    format = request.args.get('format')

    # Extract input features from the data dictionary
    tv_type = float(data['TV_Type'])
    tv_pixels = data['TV_Pixels']
    brand = data['Brand']

    # Map the brand and pixel values to integers
    brand_id = brand_map.get(brand, 4)  # Default value is 4 (NA) if brand is not found
    pixel_id = pixel_map.get(tv_pixels, -1)  # Default value is -1 if pixel value is not found

    # Make predictions using the model
    input_data = {
        'TV_Type': [tv_type],
        'TV_Pixels': [pixel_id],
        'Brand': [brand_id]
    }

    # Add headers for the model input
    model_input_headers = ['TV_Type', 'TV_Pixels', 'Brand']

    # Create a numpy array with the correct headers
    model_input_array = np.array([input_data[header][0] for header in model_input_headers])

    prediction = model.predict([model_input_array])
    print(prediction)
    print(prediction.dtype)
    # Round the prediction to two decimal places
    output = prediction.tolist()
    print(output)
    # Return the predictions as a JSON response
    return jsonify({'predict_price': output, 'input_data': data})

if __name__ == '__main__':
    app.run(debug=True)
