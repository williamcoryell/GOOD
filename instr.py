from transformers import (
    DataCollatorForSeq2Seq,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    T5ForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Trainer,
    Seq2SeqTrainer,
)
import findfile
import pickle
import json
from datasets import DatasetDict, Dataset
import csv
import xml.etree.ElementTree as ET

def read_json(data_path, data_type="train"):
    data = []

    files = findfile.find_files(data_path, [data_type, ".jsonl"], exclude_key=[".txt"])
    for f in files:
        with open(f, "r") as fin:
            for line in fin:
                data.append(json.loads(line))
    return data


class InstructDatasetLoader:
    def __init__(
        self,
        train_df_id,
        test_df_id,
        sample_size=1,
    ):
        self.train_df_id = train_df_id.sample(frac=sample_size, random_state=1999)
        self.test_df_id = test_df_id

    def create_datasets(self, tokenize_function):
        #Create the training and test dataset as huggingface datasets format.
        trains = Dataset.from_pandas(self.train_df_id)
        tests = Dataset.from_pandas(self.test_df_id)
        indomain_dataset = DatasetDict(
            {
                "train": trains,
                "test": tests,
            }
        )
        indomain_tokenized_datasets = indomain_dataset.map(
            tokenize_function, batched=True
        )

        return indomain_tokenized_datasets


def prepare_input(textInput, aspects):
    return (
        textInput + f"| aspects: {aspects}"
    )


def processPQRADF():
    lst = []
    with open('data\\Processed_Queries_and_Reviews_Annotated_Data_Final.csv', newline='', encoding="utf-8") as csvfile:
        spamreader = csv.reader(csvfile, quotechar='|')
        for row in spamreader:
            lst.append(row)
    for i in range(len(lst)):
        removeLst = []
        for j in range(len(lst[i])):
            if "[" in lst[i][j] or "]" in lst[i][j] or "GPT" in lst[i][j] or lst[i][j].isnumeric():
                removeLst.append(j)
        if removeLst:
            for j in removeLst[::-1]:
                # print(lst[i])
                # print(j)
                del lst[i][j]

    for i in lst:
        removeLst = []
        for j in range(len(i)):
            if j != 0 and i[j].count("-") < 2:
                removeLst.append(j)
        for j in removeLst[::-1]:
            del i[j]
    for i in lst:
        for j in i:
            if '"' in j:
                j = j.replace('"', "")

    newLst = []

    for i in lst:
        for j in range(len(i)):
            if j == 0:
                continue
            totLst = i[j].split("-")
            if len(totLst) < 2:
                continue
            aspect = totLst[0].lower().replace('"', '')
            if aspect[0] == " ":
                aspect = aspect[1:]
            if aspect == "general":
                continue
            label = totLst[2].lower().replace('"', '').replace(" ", "")
            if label != "positive" and label != "negative" and label != "neutral":
                continue
            try:
                if int(i[0]):
                    print(i)
            except:
                x=2
            finDict = {"text": i[0] + "| aspects: " + aspect, "labels": label}
            if finDict not in newLst:
                newLst.append(finDict)
    return newLst


def processXMLFile():
    lst = []
    tree = ET.parse('data\\train.xml')
    root = tree.getroot()
    for sentence in root.findall('sentence'):
        # print(sentence.find("text").text)
        acs = sentence.find("aspectCategories")
        sentText = sentence.find("text").text
        for aspect in acs.findall('aspectCategory'):
            # print(aspect.get("category"), aspect.get("polarity"))
            sentAspect = aspect.get("category")
            sentLabel = aspect.get("polarity")
            if sentAspect == "miscellaneous":
                continue
            lst.append({"text": sentText + "| aspects: " + sentAspect, "labels": sentLabel})
    return lst

def processRawFile():
    lst = []
    with open("data\\train.raw", "r", encoding="utf-8") as tex:
        for i, ch in enumerate(tex):
            chs = ch.replace('\n', '')
            if i % 3 == 0:
                txt = chs
            elif i%3 == 1:
                aspect = chs
            else:
                # print(txt)
                # print(aspect)
                # print(chs)
                if chs == "-1":
                    label = "negative"
                elif chs == "0":
                    label = "neutral"
                elif chs == "1":
                    label = "positive"
                lst.append({"text": txt.replace("$T$", aspect) + "| aspects: " + aspect, "label": label})

    return lst