from flask import Flask, render_template, jsonify
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

@app.route('/')
def home(blarg=None):
  return render_template('home.html', blarg='yes')

@app.route('/api/digit', methods=['POST'])
def getDigit(image=None):
  return jsonify({
    'header1': 'asdf',
    'another': 'testing'
  })