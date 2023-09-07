# Information summary module, Data labeling module, Data refine module using chatGPT API
import os
import pdb

import openai

base_path = os.path.abspath(__file__).split('/')
#base_path[-2], base_path[-1] = "data", "news1"


def Kor2Eng(text: str) -> str:
    role = "You are a role with a great talent for translating English into Korean."
    My_OpenAI_key = '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'
    openai.api_key = My_OpenAI_key
    translate = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
                {"role": "system", "content": role},
                {"role": "user", "content": text},
            ]
    )
    return translate.choices[0]['message']['content']



def labeling_module(news):
    global base_path
    # This module needs two chatGPTs model assigned a specific role.
    role1 = "You are a role with a great talent for translating Korean into English."
    role2 = """From now on, your role is to be good at summarizing sentences and grasping the emotions of the text.
          It is a role to know whether the article is positive or negative, and also what topic the article is about.
            If I give you a sentence, I want you to give me an answer in the format below.
            form
            summary:
            Positive/Negative:
            category:"""
    
    # Get the news data (sentences)
    path = "/".join(base_path[:-3]+["data","news.txt"])
    news = ""
    with open(path, 'r') as file:
        news = file.read()

    My_OpenAI_key = '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'
    openai.api_key = My_OpenAI_key

    print(f"Translating the news data to English ...")
    translate = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
                {"role": "system", "content": role1},
                {"role": "user", "content": news},
            ]
    )
    t_result = translate.choices[0]['message']['content']

    print(f"labeling the news data ...")
    label = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": role2},
            {"role": "user", "content": t_result}
        ]
    )    
    print("="*30)
    print("Summary & Labeling result: ")
    print("="*30)
    print(label)
    #pdb.set_trace()
    answer = label.choices[0]['message']['content'].split("\n\n")
    pos_neg = answer[1].split(":")[1].strip()
    category = answer[2].split(":")[1].strip()
    summary = answer[0].split(":")[1].strip()
    return summary


def Refine_module(answer: str) -> str:
    role = """You make sentences more sophisticated and smooth, and are good at paraphrasing. 
    And although the meaning is the same, try to keep the sentences as long as possible."""
    
    refined_answer = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
                {"role": "system", "content": role},
                {"role": "user", "content": answer},
            ]
    )
    return refined_answer.choices[0]['message']['content']