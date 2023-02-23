import json
import pandas as pd
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

# load dataframe
tags = [
    "api",
    "database",
    "sql",
    "nosql",
    "frontend",
    "backend",
    "docker",
    "pipeline",
    "devops",
    "sre",
    "model",
    "deployment",
    "mobile",
    "cloud",
    "server",
    "ml",
]

docs = []
for tag in tags:
    with open(f"data_error/{tag}.json", "r") as r:
        data = json.load(r)
    for d in data:
        owner = {f"owner_{k}": d["owner"][k] for k in d["owner"]}
        docs.append(
            {"group": tag, **{k: d[k] for k in d if k not in ["owner"]}, **owner}
        )
df = pd.DataFrame(docs)


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


corpus = (
    df["title"].str.lower().apply(preprocess).map(lambda x: re.sub("[;&,\.!?]", "", x))
)
df["text_clean"] = corpus

corpus_detail = df["tags"].apply(lambda x: ". ".join(x) if type(x) == list else "")
df["corpus_detail"] = corpus_detail

corpus = corpus_detail + ". " + corpus
df["corpus"] = corpus
# [title, text_clean, corpus]


# find stack group
def find_stack_group(group):
    stack_group = {
        "backend": ["api", "backend"],
        "database": ["database", "sql", "nosql", "pipeline", "model", "ml"],
        "frontend": ["frontend"],
        "devops": ["docker", "devops", "sre", "deployment", "cloud", "server"],
        "mobile": ["mobile"],
    }
    for stack in stack_group:
        if group in stack_group[stack]:
            return stack
    return ""


df["stack_group"] = df["group"].apply(find_stack_group)

# store df to json
df.to_json("preprocesed_df.json")

# find topic
topics = []
for tag in tags:
    # Sample documents
    documents = df[df["group"] == tag]["text_clean"].values.tolist()
    topic_size = 20
    if len(documents) < topic_size:
        topic_size = 5
    print(tag, len(documents))
    # Convert documents to TF-IDF representation
    tfidf_vectorizer = TfidfVectorizer()
    tfidf = tfidf_vectorizer.fit_transform(documents)

    # Perform SVD on the TF-IDF matrix
    lsa = TruncatedSVD(n_components=topic_size)
    lsa_vectors = lsa.fit_transform(tfidf)

    # Print the topics learned by the LSA model
    terms = tfidf_vectorizer.get_feature_names_out()
    for i, comp in enumerate(lsa.components_):
        row = {}
        row["tag"] = tag
        row["topic_num"] = i
        row["topics"] = [terms[j] for j in comp.argsort()[:-5:-1]]
        print("tag", tag, "Topic", i, ":", row["topics"])
        topics.append(row)
    print()
    # tag api Topic 0 : ['api', 'error', 'request', 'getting']
    # tag api Topic 1 : ['server', 'internal', '500', 'api']

df_topic = pd.DataFrame(topics)
# [tag, topic_num, topics]
# [api, 0, [api, error, request, getting]]

# store topic to csv
df_topic["label"] = 0
df_topic.to_csv("title_topic_label.csv", index=False)

print("now label manually in title_topic_label.csv")
print(
    {
        1: "Warning: A condition might warrant attention",
        2: "Error: Minor error, quick to fix",
        3: "Critical: A critical error, some features may not working",
        4: "Alert: Immidiate action may be required, unstable",
        5: "Emergency: The system is stop running",
    }
)
