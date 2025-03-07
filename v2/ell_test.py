import ell
from rich import print

MODEL = "llama-3.3-70b-versatile"
ell.models.groq.register(
    api_key="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"
)


@ell.complex(MODEL)
def chat_bot(message_history: list[ell.Message]):
    return [
        ell.system("You are jinx from lol."),
    ] + message_history


message_history = []
while True:
    user_input = input("You: ")
    message_history.append(ell.user(user_input))
    response = chat_bot(message_history)
    assert isinstance(response, ell.Message)
    print("Bot:", response.text)
    message_history.append(response)
