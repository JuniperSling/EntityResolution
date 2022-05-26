import torch
from model import BertClassification
from utils import load_config
import argparse
import pickle
from dataloader import TextDataset, BatchTextCall, choose_bert_type
from tqdm import tqdm
import requests
import random

def get_wiki_predictions(query, limit):
    predict_entity_id_list = [''] * limit
    predict_entity_label_list = [''] * limit
    predict_entity_description_list = [''] * limit
    try:
        response = requests.get(
            "https://www.wikidata.org/w/api.php?action=wbsearchentities&search="
            + query + "&language=en&limit=" + str(limit) + "&format=json").json()
        predict_entity_id_list = [ent['id'] for ent in response['search']]
        predict_entity_label_list = [ent['label'] for ent in response['search']]
        predict_entity_description_list = [ent['description'] for ent in response['search']]
        return predict_entity_id_list, predict_entity_label_list, predict_entity_description_list
    except:
        return predict_entity_id_list, predict_entity_label_list, predict_entity_description_list

def falcon_search(query, limits):
    my_json = {"text": query}
    wikidata_URI_List =[]
    wikidata_URI = []
    try:
        wikidata_URI_List = requests.post("https://labs.tib.eu/falcon/falcon2/api?mode=short&k=" +
                                          str(limits), json=my_json).json()
        # print(wikidata_URI_List)
        wikidata_URI = [uri[0].strip("<http://wikipedia.org/resource/>") for uri in wikidata_URI_List["entities"]]
        # wikidata_URI = '|'.join(wikidata_URI_List)
        return wikidata_URI
    except:
        wikidata_URI = []
        return wikidata_URI

def evaluate_with_candidate(path, limits):
    jump_count = 0

    end_line = 20000  # 需要测试的行数

    jump_prop = 1  # 跳过90%

    pbar = tqdm(total=end_line)

    right_generate_count = 0  # 候选成功覆盖到的条目数

    all_count = 0  # 总条目数
    true_all_count = 0  # 参与运算的总条目

    with open(path, 'rb') as f:
        while True:
            try:
                d = pickle.load(f)
                jump_count += 1
                if jump_count < 0: continue

                flag = random.random()

                # 随机跳过
                if flag > jump_prop:
                    all_count += 1
                    pbar.update(1)
                    if all_count >= end_line: break
                    continue

                target_entity_id = d['predict_entity_list'][d['label_list'].index(True)]  # target 预测ID
                predict_entity_id_list = falcon_search(d['entity'], limits)
                # 没有结果
                if len(predict_entity_id_list) == 0:
                    print("bad network")
                    all_count += 1
                    pbar.update(1)
                    if all_count >= end_line: break
                    continue

                if predict_entity_id_list.count(target_entity_id) != 0:
                    right_generate_count += 1

                all_count += 1
                true_all_count += 1
                print(str(limits) + "train-Generate Acc: " + str(right_generate_count / true_all_count))
                pbar.update(1)
                if all_count >= end_line:
                    print("Generate Accuracy: " + str(right_generate_count / true_all_count))
                    break
            except EOFError:
                break


if __name__ == '__main__':
    evaluate_with_candidate('../data/49000-125854/train.txt', 5)
