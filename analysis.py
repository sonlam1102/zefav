import json
from sklearn.metrics import f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import re


def compute_f1_hops(data, num_hop=2):
    label2idx = {
        'refutes': 0,
        'supports': 1
    }

    g = []
    p = []
    for d in data:
        if d['num_hops'] == num_hop:
            p.append(label2idx[d['predicted_label_new']])
            g.append(label2idx[d['label']])

    print(f1_score(g, p, average='macro'))
    return g, p


def compute_f1_challenge_no(data, challenge="Other"):
    label2idx = {
        'refutes': 0,
        'supports': 1
    }

    g = []
    p = []
    for d in data:
        if d['challenge'] == challenge:
            p.append(label2idx[d['predicted_label_new']])
            g.append(label2idx[d['label']])

    # print(f1_score(g, p, average='macro'))
    return f1_score(g, p, average='macro')


def compute_f1(data):
    label2idx = {
        'refutes': 0,
        'supports': 1
    }

    g = []
    p = []
    for d in data:
        # print(d['id'])
        p.append(label2idx[d['predicted_label_new']])
        g.append(label2idx[d['label']])

    print(f1_score(g, p, average='macro'))
    return g, p


def exploit_label(data):
    num_err = 0
    for d in data:
        prompt_pred = d['predicted_label']
        label = prompt_pred.split("###The answer is:")[1].strip().strip('\n')

        try:
            label_extract = re.search(r'(True|False|TRUE|FALSE)', label).group(0)
            if label_extract == 'True':
                d['predicted_label_new'] = 'supports'
            else:
                d['predicted_label_new'] = 'refutes'
        except Exception as e:
            # print(d['id'])
            # raise e
            d['predicted_label_new'] = 'supports'
            num_err = num_err + 1

    # print(num_err)

    return data, num_err


def draw_confusion(d_hover, d_feverous):
    g_hover_2, p_hover_2 = compute_f1_hops(d_hover, num_hop=2)
    g_hover_3, p_hover_3 = compute_f1_hops(d_hover, num_hop=3)
    g_hover_4, p_hover_4 = compute_f1_hops(d_hover, num_hop=4)

    g_feverous, p_feverous = compute_f1(d_feverous)

    cm_hover_2 = confusion_matrix(g_hover_2, p_hover_2, labels=[0, 1])
    cm_hover_3 = confusion_matrix(g_hover_3, p_hover_3, labels=[0, 1])
    cm_hover_4 = confusion_matrix(g_hover_4, p_hover_4, labels=[0, 1])

    cm_feverous = confusion_matrix(g_feverous, p_feverous, labels=[0, 1])
    # Define the labels and titles for the confusion matrix
    classes = ['Refutes', 'Supports']

    # fig, axs = plt.subplots(ncols=4)
    s1 = sns.heatmap(cm_hover_2, annot=True, cmap='Blues', fmt='g', xticklabels=classes, yticklabels=classes,
                     annot_kws={'size': 25})
    s1.set_xlabel("Predicted")
    s1.set_ylabel("Actual")
    s1.set_title("HoVer (2-hop)")

    plt.show()

    s2 = sns.heatmap(cm_hover_3, annot=True, cmap='Blues', fmt='g', xticklabels=classes, yticklabels=classes,
                     annot_kws={'size': 25})
    # Set the axis labels and title
    s2.set_xlabel("Predicted")
    s2.set_ylabel("Actual")
    s2.set_title("HoVer (3-hop)")
    plt.show()
    #
    s3 = sns.heatmap(cm_hover_4, annot=True, cmap='Blues', fmt='g', xticklabels=classes, yticklabels=classes,
                     annot_kws={'size': 25})
    # Set the axis labels and title
    s3.set_xlabel("Predicted")
    s3.set_ylabel("Actual")
    s3.set_title("HoVer (4-hop)")
    plt.show()
    #
    s4 = sns.heatmap(cm_feverous, annot=True, cmap='Blues', fmt='g', xticklabels=classes, yticklabels=classes,
                     annot_kws={'size': 25})
    # Set the axis labels and title
    s4.set_xlabel("Predicted")
    s4.set_ylabel("Actual")
    s4.set_title("FEVEROUS-S")
    plt.show()


if __name__ == '__main__':
    with open('./dump/saved/best/analysis/has_context/full/output_prediction_prompt_dev_hover.json', 'r') as f:
        hover = json.load(f)
        d_hover, _ = exploit_label(hover)
    f.close()

    with open('./dump/saved/best/analysis/has_context/full/output_prediction_prompt_dev_feverous.json', 'r') as f:
        feverous = json.load(f)
        d_feverous, _ = exploit_label(feverous)
    f.close()

    draw_confusion(d_hover, d_feverous)

    # lst_challenge = []
    # for d in d_feverous:
    #     lst_challenge.append(d['challenge'])
    # lst_challenge = list(set(lst_challenge))
    #
    # for c in lst_challenge:
    #     print("{}: {}".format(c, compute_f1_challenge_no(d_feverous, c)*100))

    # err_hover = []
    # for dh in d_hover:
    #     if dh['predicted_label_new'] != dh['label'] and dh['num_hops'] == 4:
    #         err_hover.append(dh)
    #
    # print(len(err_hover))
    #
    # with open('error_hover.json', 'w') as f:
    #     json.dump(err_hover, f, ensure_ascii=False, indent=4)
    #
    # err_feverous = []
    # for df in d_feverous:
    #     if df['predicted_label_new'] != df['label'] and df['challenge'] == 'Numerical Reasoning':
    #         err_feverous.append(df)
    #
    # print(len(err_feverous))
    #
    # with open('error_feverous.json', 'w') as f:
    #     json.dump(err_feverous, f, ensure_ascii=False, indent=4)

    # err_feverous = []
    # for df in d_feverous:
    #     if df['challenge'] == 'Numerical Reasoning':
    #         err_feverous.append(df)

    # print(len(err_feverous))
    # de_feverous, _ = exploit_label(err_feverous)
    # ge_feverous, pe_feverous = compute_f1(de_feverous)
    # cme_feverous = confusion_matrix(ge_feverous, pe_feverous, labels=[0, 1])

    # s4 = sns.heatmap(cme_feverous, annot=True, cmap='Blues', fmt='g', xticklabels=['Refutes', 'Supports'], yticklabels=['Refutes', 'Supports'],
    #                  annot_kws={'size': 25})
    # # Set the axis labels and title
    # s4.set_xlabel("Predicted")
    # s4.set_ylabel("Actual")
    # s4.set_title("FEVEROUS-S")
    # plt.show()
