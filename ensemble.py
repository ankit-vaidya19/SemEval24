import pandas as pd
import numpy as np
from itertools import combinations


e2i = {
    "disgust": 0,
    "contempt": 1,
    "anger": 2,
    "neutral": 3,
    "joy": 4,
    "sadness": 5,
    "fear": 6,
    "surprise": 7,
}
i2e = {v: k for k, v in e2i.items()}


def get_max_from_dict(model1, model2, model3, f1dict):
    if f1dict[model1] > f1dict[model2] and f1dict[model1] > f1dict[model3]:
        return model1
    elif f1dict[model2] > f1dict[model1] and f1dict[model2] > f1dict[model3]:
        return model2
    else:
        return model3


def get_outputs(model1, model2, model3, f1dict, length=1580):
    y_pred = []

    for i in range(length):
        temp = np.zeros((8), dtype=int)
        temp[e2i[str(model1[i][0])]] += 1
        temp[e2i[str(model2[i][0])]] += 1
        temp[e2i[str(model3[i][0])]] += 1
        if np.max(temp) == 1:
            max_model = get_max_from_dict(model1, model2, model3, f1dict)
            y_pred.append(str(max_model[i][0]))
        else:
            cur = np.argmax(temp)
            y_pred.append(i2e[(cur)])
    return y_pred


def create_new_submission(models):
    possible_combinations = []
    possible_combinations = list(combinations(models, 3))
    counter = 1
    for combination in possible_combinations:
        y_pred = get_outputs(combination[0], combination[1], combination[2], f1_scores)
        finalarr = np.array(y_pred)
        submission = pd.DataFrame(finalarr, columns=["label"])
        submission.to_csv(f"combination_{counter}.csv", index=False)
        print(f"combination_{counter} created")
        counter += 1


another_mbert = np.array(
    pd.read_csv(
        "/kaggle/input/final-ensemble-files/predicted_emotions_another_mbert.csv"
    )
)
hing_bert_lid = np.array(
    pd.read_csv(
        "/kaggle/input/final-ensemble-files/predicted_emotions_hing_bert_lid.csv"
    )
)
hing_bert = np.array(
    pd.read_csv(
        "/kaggle/input/final-ensemble-files/predicted_emotions_l3cube_hing_bert.csv"
    )
)
mbert = np.array(
    pd.read_csv("/kaggle/input/final-ensemble-files/predicted_emotions_mbert.csv")
)
multilingual_bert_uncased = np.array(
    pd.read_csv(
        "/kaggle/input/final-ensemble-files/predicted_emotions_multilingual_bert_uncased.csv"
    )
)
nirantk = np.array(
    pd.read_csv("/kaggle/input/final-ensemble-files/predicted_emotions_nirantk.csv")
)
roberta = np.array(
    pd.read_csv("/kaggle/input/final-ensemble-files/predicted_emotions_roberta.csv")
)

another_mbert = tuple(map(tuple, another_mbert))
hing_bert_lid = tuple(map(tuple, hing_bert_lid))
hing_bert = tuple(map(tuple, hing_bert))
mbert = tuple(map(tuple, mbert))
multilingual_bert_uncased = tuple(map(tuple, multilingual_bert_uncased))
nirantk = tuple(map(tuple, nirantk))
roberta = tuple(map(tuple, roberta))

f1_scores = {
    another_mbert: 0.3543077022,
    hing_bert_lid: 0.2822756604,
    hing_bert: 0.3166413358,
    mbert: 0.349018775,
    multilingual_bert_uncased: 0.3044980282,
    nirantk: 0.3281418016,
    roberta: 0.3138602062,
}


models = [
    another_mbert,
    hing_bert_lid,
    hing_bert,
    mbert,
    multilingual_bert_uncased,
    nirantk,
    roberta,
]


create_new_submission(models)
