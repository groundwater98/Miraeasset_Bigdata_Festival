from models import chatGPT, predict, recommend
from typing import Tuple
import openai
import pdb


def get_menu() -> Tuple[str, str, str]:
    # 사용자에게서 어떤 서비스를 원하는지 가져온다.
    # chatGPT를 이용해서 대화형에 가깝게 만든다.
    My_OpenAI_key = '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'
    openai.api_key = My_OpenAI_key

    print(f"Hi I'm Sacretary for your Investment.\n What can I help you?")
    print(f"=========Here is services we provide=========")
    print(f"1. Information Summary\n2. Prediction\n3. Recommendation\n4. Quit")
    print(f"=============================================\n")
    print(f"You can also give the sentence.")
    question = input(f"Please write the service name, number you want or the sentence. ")

    role = """You have great understanding and are good at finding out what the other person wants from what they say. 
                If I give you a sentence, you will have to tell me which of the services below you would like. 
                Number 1 is for news information summary, number 2 is stock recommendation, and number 3 is stock price prediction. 
                The answer should be as follows:
                
                Service: News information summary, stock recommendation, stock price prediction
                
                answer:
                service:
                Information needed: Explain very specifically what information you need, giving examples."""
    # 서비스가 불명확하면, 재질문.
    print(f"Analyzing what you want ...\n")
    # messages는 대화의 흐름을 기억할 수 있게 해준다.
    # system, user, assistant, user, assistant ... 반복해나가면서 대화 흐름 기억 가능.
    # role은 총 3가지가 있다. system, user, assistant <-- 이건 스펠링이 확실히 맞는지 모름 
    # system은 chatGPT의 역할 부여, user는 우리의 질문, assistant는 chatGPT의 답변
    answer = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
                {"role": "system", "content": role},
                {"role": "user", "content": question},
            ]
    )
    answer = answer.choices[0]['message']['content']
    print(answer)
    idx = answer.find("Information needed")
    service = answer[8:idx].strip() # 고객이 원하는 서비스
    information_needed = answer[idx+19:].strip() # 서비스 제공을 위해 필요한 정보에 대한 질문
    #pdb.set_trace()
    return service, information_needed, question


def kakao():
    while True:
        service, information_needed, question = get_menu()
        if service == 'Information Summary':
            print(f"Labeling ...")
            chatGPT.labeling_module() 
        elif service == 'Stock price prediction':
            role = """You will receive a statement asking for a stock price prediction, 
            and your role is specialized in finding out which stock you want to predict. 
            Please answer in the first format below. If the forecast period and stock are not specified, 
            please answer in the second format.

            First answer format:
            Predicted stocks:
            Forecast Period: only days

            Second response format:
            There is insufficient information to satisfy your needs. Please tell me specifically what stock price you want and when.
            Example) Predict Apple's stock price tomorrow."""

            answer = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                        {"role": "system", "content": role},
                        {"role": "user", "content": question},
                    ]
            )
            answer = answer.choices[0]['message']['content']
            if answer.find("First answer format") == -1:
                # 첫번째 대답 형식 아닌 경우: 예측 종목도 없고, 기간도 없는 경우
                pdb.set_trace()
                answer = answer.split("\n")
                stock = answer[1].split(":")[0]
                period = int(answer[1].split(":")[1].split()[0])
            else:
                # 두번째 대답 형식 아닌 경우: 예측, 기간 특정된 경우
                pass
            predict.predict(stock, period)
        elif service == 'Stock recommendation':
            user_inform = "우리 서비스 이용중인 고객의 정보를 불러와야 함."
            recommend.recommend(user_inform)
        else:
            print(f"\n\nPlease write the correct service!!")

if __name__ == '__main__':
    kakao()
    print(f"\n\nThank you for your using.")