import openai

My_OpenAI_key = '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'

openai.api_key = My_OpenAI_key

"""
response = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[
        {"role": "system", "content": "You are a smart, very knowledgeable facilitator about the economy."},
        {"role": "user", "content": "Where is the Korea?"},
        #{"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
        #{"role": "user", "content": "Where was it played?"}
    ]
)
"""
"""
response = openai.Completion.create(
    model="text-davinci-003",
    prompt="안녕, 내 이름은 이수진이야. \n\nQ: 이름이 뭘까?\nA:",
    temperature=0,
    max_tokens=100,
    top_p=1,
    frequency_penalty=0.0,
    presence_penalty=0.0,
    stop=["\n"]
)
"""
question1 = "Is it worth investing in Apple? I have $100,000, and I usually do about 45 stock trades a week."
question2 = "Is it worth investing in Apple?"
question3 = "where is the korea?"

# 데이터 1000개로 학습
response1 = openai.Completion.create(
    model="davinci:ft-personal-2023-07-07-07-12-37",
    #prompt="Q: What's the Apple stock outlook?",
    prompt=question3,
    temperature=0,
    max_tokens=100,
    top_p=1,
    frequency_penalty=0.0,
    presence_penalty=0.0,
    #stop=["\n"]
)

# 데이터 100개로 학습
response2 = openai.Completion.create(
    model="davinci:ft-personal-2023-07-09-13-13-18",
    #prompt="Q: What's the Apple stock outlook?",
    prompt=question3,
    temperature=0,
    max_tokens=100,
    top_p=1,
    frequency_penalty=0.0,
    presence_penalty=0.0,
    #stop=["\n"]
)

# 데이터 10개로 학습
response3 = openai.Completion.create(
    model="davinci:ft-personal-2023-07-11-11-10-36",
    #prompt="Q: What's the Apple stock outlook?",
    prompt=question3,
    temperature=0,
    max_tokens=100,
    top_p=1,
    frequency_penalty=0.0,
    presence_penalty=0.0,
    #stop=["\n"]
)

# openai 제공 모델
response4 = openai.Completion.create(
    model="text-davinci-003",
    #prompt="Q: What's the Apple stock outlook?",
    prompt=question3,
    temperature=0,
    max_tokens=100,
    top_p=1,
    frequency_penalty=0.0,
    presence_penalty=0.0,
    #stop=["\n"]
)
print(response1['choices'][0]['text'])
print("="*40)
print(response2['choices'][0]['text'])
print("="*40)
print(response3['choices'][0]['text'])
print("="*40)
print(response4['choices'][0]['text'])
#print(f"The answer: {response.choices[0]['message']['content']}")
