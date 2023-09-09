from flask import Flask, jsonify, request
import requests, sys, json

import service

application = Flask(__name__)
a = {}
# 우분투 연결 키
# A0.451cbe76-d158-4071-a9b0-7c150217826b.veMpQ9iceYBFDl2VQdOzNzO-cEu4QD5Qzg
# 구름 연결 키
# A0.32bc5757-980c-49ac-8a16-216879c1e5c5.ggThWy5B_6PCsUL1R9enYBEuOPuQhBpaHw

@application.route('/')
def hello_world():
    print('hello_world')
    return 'Hello, World!'

@application.route("/webhook/", methods=["POST"])
def webhook():
    global a
    #print("여기서 문제다.")
    request_data = json.loads(request.get_data())
    #print('hello')
    #print(request_data)
    question = request_data['action']['params']['question']
    answer = service.kakao(question)
    a[request_data['userRequest']['user']['id']] = answer
    #a[request_data['user']] = request_data['result']['choices'][0]['message']['content']
    return 'OK'

@application.route("/question", methods=["POST"])
def get_question():
    global a
    #print("get_question")
    request_data = json.loads(request.get_data())
    #print(request_data)
    response = { "version": "2.0", "template": { "outputs": [{
        "simpleText": {"text": f"질문을 받았습니다. AI에게 물어보고 올께요!: {request_data['action']['params']['question']}"}
    }]}}
    a[request_data['userRequest']['user']['id']] = '아직 AI가 처리중이에요'
    url = "https://kakao-chatbot-suabn.run.goorm.site/webhook/"
    data = json.dumps(request_data)
    #print('before post to webhook')
    #print(data)
    requests.post(url, data)
    return jsonify(response)

@application.route("/ans", methods=["POST"])
def hello2():
    #print("hello2")
    request_data = json.loads(request.get_data())
    #print(request_data)
    response = { "version": "2.0", "template": { "outputs": [{
        "simpleText": {"text": f"답변: {a.get(request_data['userRequest']['user']['id'], '질문을 하신적이 없어보여요. 질문부터 해주세요')}"}
    }]}}
    return jsonify(response)

if __name__ == "__main__":
    application.run(host="0.0.0.0",port=80,debug=True)