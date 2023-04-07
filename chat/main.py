from chatbot import Chatbot
from config import STARTING_DIALOG


def main():
    print("Preparing the chatbot...")
    chatbot = Chatbot(STARTING_DIALOG)
    print("\nHedgehog: " + STARTING_DIALOG)

    while True:
        user_input = input("User: ")
        response = chatbot.chat(user_input)
        print("\nHedgehog: " + response)


if __name__ == "__main__":
    main()
