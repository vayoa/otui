import ell
from rich import print

# ell.init(verbose=True)

MODEL = "llama-3.1-70b-versatile"
ell.models.groq.register(
    api_key="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"
)


# def get_random_adjective():
#     adjectives = ["enthusiastic", "cheerful", "warm", "friendly"]
#     return random.choice(adjectives)


# @ell.simple(model=MODEL)
# def hello(name: str):
#     """You are a helpful assistant."""
#     adjective = get_random_adjective()
#     return f"Say a {adjective} hello to {name}!"


# greeting = hello("Sam Altman")
# print(greeting)


@ell.complex(model=MODEL, temperature=0.7)
def chat_bot(message_history: list[ell.Message]):
    return [
        ell.system("You are a friendly chatbot. Engage in casual conversation."),
    ] + message_history


message_history = []
while True:
    user_input = input("You: ")
    message_history.append(ell.user(user_input))
    response = chat_bot(message_history)
    print("Bot:", response.text)
    message_history.append(response)
