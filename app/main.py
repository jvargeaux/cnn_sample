from flask import Flask, render_template, request, jsonify
from flask_cors import CORS, cross_origin
from dotenv import load_dotenv
from predict import DigitClassifierCNNEvaluator


load_dotenv()
app = Flask(__name__, static_url_path='/app/static')
CORS(app)


def is_allowed_filetype(filename):
  ALLOWED_FILETYPES = {'png'}
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
    evaluator = DigitClassifierCNNEvaluator()
    tensor = evaluator.transform_image(img_bytes)
    prediction, percentages = evaluator.predict(tensor)
    return jsonify({
      'prediction': prediction.item(),
      'percentages': percentages
    })
  
  except Exception as e:
    print(e)
    return jsonify({
      'error': 'Problem reading file.'
    })
  

if __name__ == '__main__':
  app.run(host='127.0.0.1', port=5000)