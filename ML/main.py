from models import chatGPT, prediction, recommend


def get_menu() -> str:
    print(f"Hi I'm Sacretary for your Investment.\n What can I help you?")
    print(f"=========Here is services we provide=========")
    print(f"1. Information Summary\n2. Prediction\n3. Recommendation\n4. Quit")
    print(f"=============================================\n")
    selection = input(f"Please write the service name or number you want. ")
    return selection


def kakao():
    while True:
        selection = get_menu()
        if selection in ['Information Summary', 'Information', '1']:
            print(f"Labeling ...")
            chatGPT.labeling_module() 
        elif selection in ['Prediction', '2']:
            stock = input(f"Which stock price do you want to know? ")
            prediction.predict(stock)
        elif selection in ['Recommend', '3']:
            pass
        elif selection in ['Quit', 'QUIT', '4']:
            break
        else:
            print(f"\n\nPlease write the correct service!!")

if __name__ == '__main__':
    kakao()
    print(f"\n\nThank you for your using.")