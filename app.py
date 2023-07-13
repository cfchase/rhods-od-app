import os
import json
from flask import Flask, jsonify, request
from prediction import predict

app = Flask(__name__)

@app.route('/')
@app.route('/api/status')
def status():
    return jsonify({'status': 'ok'})

@app.route('/api/predictions', methods=['POST'])
def create_prediction():
    data = request.data or '{}'
    body = json.loads(data)
    return jsonify(predict(body))


if __name__ == '__main__':
    port = os.environ.get('FLASK_PORT') or 8080
    port = int(port)

    app.run(port=port,host='0.0.0.0')
