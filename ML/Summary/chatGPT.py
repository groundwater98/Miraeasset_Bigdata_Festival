# Information summary module, Data labeling module, Data refine module using chatGPT API
import os

import openai

base_path = os.path.abspath(__file__).split('/')
#base_path[-2], base_path[-1] = "data", "news1"


def labeling_module():
    global base_path
    # This module needs two chatGPTs model assigned a specific role.
    role1 = "You are a role with a great talent for translating Korean into English."
    role2 = "From now on, your role is to be good at summarizing sentences and grasping the emotions of the text.\
          It is a role to know whether the article is positive or negative, and also what topic the article is about.\
            If I give you a sentence, I want you to give me an answer in the format below.\
            form\
            summary:\
            Positive/Negative:\
            category:"
    
    # Get the news data (sentences)
    path = "/root/workspace/QA/data/test_data/news1.txt"
    news = ""
    with open(path, 'r') as file:
        news = file.read()

    My_OpenAI_key = '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'
    openai.api_key = My_OpenAI_key

    print(f"Translating the news data to English ...")
    translate = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
                {"role": "system", "content": role1},
                {"role": "user", "content": news},
            ]
    )
    t_result = translate.choices[0]['message']['content']

    print(f"labeling the news data ...")
    label = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": role2},
            {"role": "user", "content": t_result}
        ]
    )    
    print(f"The label\n================================\
           {label.choices[0]['message']['content']}\n\n")



def Inform_module():
    pass


def Refine_module(data: str) -> str:
    role = "From now on, you are a role with a lot of knowledge about the language. \
        In the future, I will give you a sentence, and you can correct the awkwardness or flow of the sentence."
    
    refined_answer = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
                {"role": "system", "content": role},
                {"role": "user", "content": news},
            ]
    )
    return refined_answer.choices[0]['message']['content']