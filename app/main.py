from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from predict import transform_image, get_prediction


load_dotenv()
app = Flask(__name__)


ALLOWED_FILETYPES = {'png'}
def is_allowed_filetype(filename):
  return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_FILETYPES


@app.route('/')
def home(blarg=None):
  return render_template('home.html', blarg='yes')


@app.route('/api/digit', methods=['POST'])
def getDigit(image=None):
  # Get image file
  file = request.files.get('file')

  # If empty or not png, return error
  if file is None or file.filename == '':
    return jsonify({
      'error': 'No file provided.'
    })
  if not is_allowed_filetype(file.filename):
    return jsonify({
      'error': 'Filetype not supported.'
    })
  
  # Transform image bytes to tensor, and pass to model
  try:
    img_bytes = file.read()
    tensor = transform_image(img_bytes)
    prediction, percentages = get_prediction(tensor)
    return jsonify({
      'prediction': prediction.item(),
      'percentages': percentages
    })
  except Exception as e:
    print(e)
    return jsonify({
      'error': 'Problem reading file.'
    })