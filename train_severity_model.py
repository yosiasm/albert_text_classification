import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AlbertTokenizer, AlbertModel
import pandas as pd
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

TARGET_COL = "label"
FEATURE_COL = "text_clean"
ENCODE_SIZE = 64
NUM_EPOCHS = 5
LEARNING_RATE = 2e-5

# load df
df = pd.read_json("labeled_data.json")
df[[TARGET_COL]] = df[[TARGET_COL]].apply(
    lambda col: pd.Categorical(col).codes)

# split data train and data test
trains = []
for target in df[TARGET_COL].unique():
    trains.append(df[df[TARGET_COL] == target].sample(
        frac=0.8, random_state=200))
train = pd.concat(trains)
test = df.drop(train.index)

# store data test for eval
test.to_json(f"test_{TARGET_COL}_df.json")

# Define device to compute
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

# initialize the tokenizer
tokenizer = AlbertTokenizer.from_pretrained("albert-base-v2")

# Convert the text data to input tensors
input_ids = []
attention_masks = []
for text in train[FEATURE_COL]:
    encoded_dict = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=ENCODE_SIZE,
        pad_to_max_length=True,
        return_attention_mask=True,
    )
    input_ids.append(encoded_dict["input_ids"])
    attention_masks.append(encoded_dict["attention_mask"])

# put into gpu
input_ids = torch.tensor(input_ids, dtype=torch.long).to("cuda")
attention_masks = torch.tensor(attention_masks, dtype=torch.long).to("cuda")
labels = torch.tensor(train[TARGET_COL].values, dtype=torch.long).to("cuda")

# create the data loader for training and validation
train_dataset = TensorDataset(input_ids, attention_masks, labels)
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# initialize the ALBERT model from scratch
model: AlbertModel = AlbertModel.from_pretrained("albert-base-v2")
model.classifier = torch.nn.Linear(
    in_features=model.config.hidden_size, out_features=len(
        set(train[TARGET_COL]))
)
model.to("cuda")

# define the loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


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


# train the model
print("training step")
for epoch in range(NUM_EPOCHS):
    # set the model to training mode
    print("epoch", epoch)
    model.train()

    # loop over the training data
    for step, batch in tqdm(enumerate(train_dataloader)):
        b_input_ids = batch[0].to(device)
        b_attention_masks = batch[1].to(device)
        b_labels = batch[2].to(device)

        # zero the gradients
        optimizer.zero_grad()

        # forward pass
        outputs = model(b_input_ids, attention_mask=b_attention_masks)
        logits = model.classifier(outputs[0][:, 0, :])
        loss = criterion(logits, b_labels)

        # backward pass
        loss.backward()
        optimizer.step()

    # set the model to evaluation mode
    test["predict"] = test[FEATURE_COL].apply(
        classify, model=model, tokenizer=tokenizer
    )
    res = test.apply(lambda x: 1 if x[TARGET_COL]
                     == x["predict"] else 0, axis=1).sum()
    print(res, "/", len(test), res / len(test))

    # store model
    torch.save(model.state_dict(), f"{TARGET_COL}_model{epoch}.pt")
