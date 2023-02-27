import  torch
from transformers import BertTokenizer


class EeArgs:
    tasks = ["ner"]
    data_name = "duee"
    data_dir = "ee"
    bert_dir = "model_hub/chinese-bert-wwm-ext/"
    save_dir = "./checkpoints/{}/{}_{}_model.pt".format(data_dir, tasks[0], data_name)
    train_path = "./data/{}/{}/duee_train.json".format(data_dir, data_name)
    dev_path = "./data/{}/{}/duee_dev.json".format(data_dir, data_name)
    test_path = "./data/{}/{}/duee_dev.json".format(data_dir, data_name)
    label_path = "./data/{}/{}/labels.txt".format(data_dir, data_name)
    with open(label_path, "r",encoding="UTF-8") as fp:
        entity_label = fp.read().strip().split("\n")
    ent_label2id = {}
    ent_id2label = {}
    for i, label in enumerate(entity_label):
        ent_label2id[label] = i
        ent_id2label[i] = label
    ner_num_labels = len(entity_label)
    train_epoch = 20
    train_batch_size = 32
    eval_batch_size = 32
    eval_step = 500
    max_seq_len = 256
    weight_decay = 0.01
    adam_epsilon = 1e-8
    max_grad_norm = 5.0
    lr = 3e-5
    other_lr = 3e-4
    warmup_proportion = 0.01
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained(bert_dir)