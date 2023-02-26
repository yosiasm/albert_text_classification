import torch
from transformers import AlbertTokenizer, AlbertModel

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# LOAD MODEL
# initialize the tokenizer
tokenizer = AlbertTokenizer.from_pretrained("albert-base-v2")

# classify text


def classify(text, model, tokenizer):
    # Define the input sequence
    input_text = text

    # Tokenize the input sequence
    input_ids = torch.tensor(
        [tokenizer.encode(input_text, add_special_tokens=True)])

    # Convert the input sequence to a tensor on the GPU (if you're using a GPU)
    input_ids = input_ids.to(
        "cuda") if torch.cuda.is_available() else input_ids

    # Make the prediction
    with torch.no_grad():
        outputs = model(input_ids)
        hidden_state = outputs[0][:, 0, :]
        logits = model.classifier(hidden_state)
        probs = torch.nn.functional.softmax(logits, dim=1)
        _, predicted_class = torch.max(probs, dim=1)

    # Print the predicted class
    # print(f'Predicted class: {predicted_class}')
    return predicted_class[0].item()


def load_model(path):
    # Initialize the ALBERT model from scratch
    model = AlbertModel.from_pretrained("albert-base-v2")
    model.classifier = torch.nn.Linear(
        in_features=model.config.hidden_size, out_features=5
    )
    model.to("cuda")

    # Load the saved model
    model.load_state_dict(torch.load(path))
    return model


SEVERITY_MODEL = load_model('severity_model4.pt')
STACK_GROUP_MODEL = load_model('stack_group_model4.pt')


# SETUP API
class BodyRequest(BaseModel):
    text: str


class BodyResponse(BaseModel):
    severity: str
    stack_group: str


app = FastAPI()


@app.get('/', response_class=HTMLResponse)
async def root():
    return '<h1>😪</h1>'


@app.post("/predict/")
async def create_item(body_request: BodyRequest):
    # preprocess
    text = body_request.text
    text = preprocess(text)

    # classify severity
    severity_code = classify(
        text=body_request.text,
        model=SEVERITY_MODEL,
        tokenizer=tokenizer
    )
    # classify stack group
    stack_group_code = classify(
        text=body_request.text,
        model=STACK_GROUP_MODEL,
        tokenizer=tokenizer
    )
    severity = ['warning', 'error', 'alert',
                'critical', 'emergency'][severity_code]
    stack_group = ["backend",
                   "database",
                   "frontend",
                   "devops",
                   "mobile"][stack_group_code]
    response = BodyResponse(stack_group=stack_group, severity=severity)

    return response

# preprocess


def preprocess(text):
    for stopword in [
        " to ",
        " in ",
        "how ",
        " the ",
        " with ",
        " of ",
        " and ",
        " is ",
        "quot",
        " on ",
        " from ",
        " for ",
        " not ",
        " using ",
        "can ",
        " when ",
        " an ",
        " do ",
        " it ",
        " by ",
        " or ",
        " after ",
        " why ",
        " as ",
        " my ",
        " that ",
        " get ",
        " into ",
        "what ",
        "where",
        "why ",
        " but ",
        " this ",
        " cannot ",
        " if ",
        "&#39",
        " can ",
    ]:
        text = text.replace(stopword, " ")
    return text
