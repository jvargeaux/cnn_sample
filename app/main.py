from flask import Flask, render_template, request, jsonify
from flask_cors import CORS, cross_origin
from dotenv import load_dotenv
from predict import transform_image, get_prediction


load_dotenv()
app = Flask(__name__, static_url_path='/app/static')
CORS(app, resources={r"/*": {"origins": "*"}})
app.config['CORS_HEADERS'] = 'Content-Type'


ALLOWED_FILETYPES = {'png'}
def is_allowed_filetype(filename):
  return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_FILETYPES


@app.route('/')
def home():
  return render_template('home.html')


@app.route('/api/digit', methods=['POST'])
@cross_origin()
def getDigit():
  # Get image file
  file = request.files.get('file')

  # If empty or not png, return error
  if file is None or file.filename == '':
    json = jsonify({
      'error': 'No file provided.'
    })
    json.headers.add('Access-Control-Allow-Origin', '*')
    return json
  if not is_allowed_filetype(file.filename):
    json = jsonify({
      'error': 'Filetype not supported.'
    })
    json.headers.add('Access-Control-Allow-Origin', '*')
    return json
  
  # Transform image bytes to tensor, and pass to model
  try:
    img_bytes = file.read()
    tensor = transform_image(img_bytes)
    prediction, percentages = get_prediction(tensor)
    json = jsonify({
      'prediction': prediction.item(),
      'percentages': percentages
    })
    json.headers.add('Access-Control-Allow-Origin', '*')
    return json
  except Exception as e:
    print(e)
    json = jsonify({
      'error': 'Problem reading file.'
    })
    json.headers.add('Access-Control-Allow-Origin', '*')
    return json