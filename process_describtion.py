import pandas as pd
import json
import spacy
import argparse


def sentences(par):
    nlp = spacy.load("en")
    return [sent.string for sent in nlp(par).sents if len(sent.string) > 1]


def media_diet():
    answers = [pd.read_csv("data/media_diet/V10.csv"), pd.read_csv("data/media_diet/V11.csv")]

    q = {
        # "group": [40],
        "family": [46, 65, 100],
        #"frog": [84, 85, 86, 87]
    }
    ethn = 40
    users = dict()
    for ans in answers:
        for idx, row in ans.iterrows():
            if idx > 1:
                history = dict()
                for col in q.keys():
                    history[col] = []
                    for val in q[col]:
                        if row[val] == "NaN" or pd.isna(row[val]) or "n/a" in row[val].lower()\
                                or row[val].lower() == "na;":
                            history[col].append("")
                        else:
                            history[col].extend(sentences(row[val]))
                if not isinstance(row[ethn], str) or row[ethn].lower()[:7] != "spanish":
                    history["group"] = 1
                else:
                    history["group"] = 0
                users[row[8]] = history

    print(len(users.keys()))
    json.dump(users, open("answers.json", "w"), indent=4)
    return users

def EBEP():
    users = dict()
    answers = pd.read_csv("data/EBEP/EBEP.csv")
    ethn = 18
    q = {
        "family": ["LangVar1", "LangVar2"],
        #"frog": []
    }

    for idx, row in answers.iterrows():
        if idx > 16:
            history = dict()
            for col in q.keys():
                history[col] = []
                for val in q[col]:
                    if row[val] == "NaN" or pd.isna(row[val]) or "n/a" in row[val].lower() or row[val].lower() == "na;":
                        history[col].append("")
                    else:
                        history[col].extend(sentences(row[val]))
            if row[ethn] == "Hispanic":
                history["group"] = 0
                users[row[8]] = history
            else:
                history["group"] = 1
                users[row[8]] = history

    json.dump(users, open("data/EBEP/answers.json", "w"), indent=4)
    return users

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="name one of the dataset, either media_diet or EBEP")
    parser.add_argument("--picture", help="family or frog?")
    args = parser.parse_args()

    if args.dataset == "EBEP":
        info = EBEP()
    else:
        info = media_diet()

    df = {"text": list(),
          "label": list()}

    for user in info.keys():
        df["text"].extend(info[user][args.picture])
        df["label"].extend([info[user]["group"] for i in range(len(info[user][args.picture]))])
    pd.DataFrame.from_dict(df).to_csv("data/" + args.dataset + "/dataset.csv")

# Hispanic = 0
# white = 1
