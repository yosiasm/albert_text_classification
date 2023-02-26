import pandas as pd

# load csv topic
df_labeled_topic = pd.read_csv("title_topic_label.csv")
# load json data
df = pd.read_json("preprocesed_df.json")


# find topic from text
def find_topic(text, topics_group):
    topics = df_labeled_topic[df_labeled_topic["tag"] == topics_group]
    selected_topic = ""
    score = 0
    for index, row in topics.iterrows():
        temp_score = 0
        topic = row["topics"]
        for t in topic:
            if t in text:
                temp_score += 1
        if temp_score >= score:
            score = temp_score
            selected_topic = row["topic_num"]
    return selected_topic


# find label from topic
def find_label(group, topic_num):
    return df_labeled_topic[
        (df_labeled_topic["tag"] == group)
        & (df_labeled_topic["topic_num"] == topic_num)
    ].iloc[0]["label"]


df_train = df[["title", "text_clean", "corpus", "group", "stack_group"]]
df_train["topic"] = df_train.apply(
    lambda x: find_topic(x["text_clean"], x["group"]), axis=1
)
df_train["severity"] = df_train.apply(
    lambda x: find_label(x["group"], x["topic"]), axis=1)
# [title, text_clean, corpus, group, topic, label]
# Fast API getting...,  fast api getting...,    python. api. sqlalche...,	api,	3,	2

# store df_train
df_train.to_json("labeled_data.json")
