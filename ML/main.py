from models import chatGPT, prediction, recommend
from typing import Tuple
import openai


def get_menu() -> Tuple(str, str):
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
    
    print(f"Analyzing what you want ...")
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
    idx = answer.find("Information needed")
    service = answer[8:idx].split(":")[1] # 고객이 원하는 서비스
    information_needed = answer[2].split(":")[1] # 서비스 제공을 위해 필요한 정보에 대한 질문
  
    return service, information_needed


def kakao():
    while True:
        service, information_needed = get_menu()
        if service == 'Information Summary':
            print(f"Labeling ...")
            chatGPT.labeling_module() 
        elif service == 'Stock price prediction':
            stock = input(f"Which stock price do you want to know? ")
            prediction.predict(stock)
        elif service == 'Stock recommendation':
            user_inform = "우리 서비스 이용중인 고객의 정보를 불러와야 함."
            recommend.recommend(user_inform)
        else:
            print(f"\n\nPlease write the correct service!!")

if __name__ == '__main__':
    kakao()
    print(f"\n\nThank you for your using.")