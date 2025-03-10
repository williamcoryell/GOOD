import autocuda
import torch
from transformers import (
    DataCollatorForSeq2Seq,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline,
    AutoTokenizer,
    Trainer,
    TrainingArguments
)
import os
import time
import pickle
from instr import *
import pandas as pd
import random
global id2label
global label2id
id2label = {0: "negative", 1: "positive"}
label2id = {"negative": 0, "positive": 1}



class DeBERTa:
    def __init__(self, checkpoint):
        if not os.path.isdir(checkpoint):
            self.tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")
            self.model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-v3-large")
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
            self.model = AutoModelForSequenceClassification.from_pretrained(checkpoint, id2label=id2label, label2id=label2id)
        self.device = autocuda.auto_cuda()
        self.model.to(self.device)
        self.pipe = pipeline("text-classification", model=self.model, tokenizer=self.tokenizer, top_k=None)
        if not os.path.isdir(checkpoint):
            self.model.save_pretrained(checkpoint)
            self.tokenizer.save_pretrained(checkpoint)

    def tokenize_inputs(self, sample):
        model_inputs = self.tokenizer(sample["text"], max_length=128, truncation=True, padding="max_length")
        newLst = []
        for i, com in enumerate(self.tokenizer(sample["labels"])["input_ids"]):
            if sample["labels"][i] == "positive":
                newLst.append([0, 1])
            elif sample["labels"][i] == "neutral":
                newLst.append([1, 1])
            elif sample["labels"][i] == "negative":
                newLst.append([1, 0])
            # print([com[1], com[2]])
            # newLst.append()
        model_inputs["labels"] = newLst
        
        return model_inputs

    def train(self, datasets, **kwargs):
        args = TrainingArguments(**kwargs)

        trainer = Trainer(
            self.model,
            args,
            train_dataset=datasets["train"],
            eval_dataset=datasets["test"],
            tokenizer=self.tokenizer,
            data_collator=DataCollatorForSeq2Seq(self.tokenizer),
        )

        torch.cuda.empty_cache()
        trainer.train()

        trainer.save_model("checkpoint.config")
        return trainer

    def predict(self, text, ATE):

        # i hate this APC stuff lol
        apc = prepare_input(text, ATE)
        pipeResult = self.pipe(apc)
        # print(apc)
        for i in pipeResult[0]:
            if i["label"] == "positive":
                positive = i
            else:
                negative = i

        results = {
            "combined": apc,
            "posScore": positive['score'],
            "negScore": negative['score'],
            "totalScore": positive['score'] - negative['score']
        }
        return results


training_args = {
    "output_dir": "checkpoints",
    "evaluation_strategy": "steps",
    "save_strategy": "steps",
    "learning_rate": 5e-5,
    "per_device_train_batch_size": 32,
    "per_device_eval_batch_size": 32,
    "num_train_epochs": 3,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "load_best_model_at_end": True,
    "push_to_hub": False,
    "eval_accumulation_steps": 64,
    "logging_steps": 64,
    "use_mps_device": False,
    'fp16': True,
    "save_steps" : 64*2
}
    


# XMLLst = processXMLFile()
# # newLst = processPQRADF()
# weirdLst = processRawFile()

# # print(len(newLst))

# # for i in range(10):
# #     print(newLst[i])

# trdf = read_json("data\\train.jsonl", "train")

# fullLst = XMLLst + trdf
# tedf = read_json("data\\test.jsonl", "test")

# random.shuffle(fullLst)

# import json

# file_path = "my_list.json"



# with open('my_list.json', 'r') as file:
#     data = json.load(file)

# while True:
#     datapoint = data[random.randint(0, len(data)-1)]
#     text = datapoint['text'].split("|")[0]
#     print(text)
#     aspect = input()
#     if not aspect:
#         break
#     data.append({"text": f"{text}| aspects: {aspect}", "label": "neutral"})



# with open(file_path, 'w') as json_file:
#     json.dump(data, json_file, indent=1)
# exit()
# tedf = pd.DataFrame(tedf)
# trdf = pd.DataFrame(fullLst)

# loader = InstructDatasetLoader(trdf, tedf)


Model = DeBERTa("checkpoint.config")

# tokenized = loader.create_datasets(
#     Model.tokenize_inputs
# )

# with open('data.pkl', 'wb') as f:
#     # Dump the data into the file
#     pickle.dump(tokenized, f)

# with open("data.pkl", "rb") as f:
#     tokenized = pickle.load(f)

# Model.train(tokenized, **training_args)







#examples for the class:

#i think that dogs are the best thing on the planet earth. I hope that every dog gets everything that it could ever want like a bunch of pets and treats.
# topic = "trump"
# urls = ap.get_urls(topic)
# lstOfStuff = []
# # random.shuffle(urls)
# for i in urls:
#     artCont = ap.get_article_content(i)
#     para = artCont.split("\n")
#     score = 0
#     addString = ""
#     scoringPara = 0
#     for j in para:
#         retStr = j
#         # print(f"addString: {addString}")
#         # print(f"retStr: {retStr}")

#         if len(retStr) < 250:
#             addString += retStr
#             continue

#         if addString:
#             retStr = addString + " " + retStr

#         # print(len(retStr))

#         addString = ""
#         guess = Model.predict(retStr, topic)["totalScore"]
#         # print(retStr)
#         # print(guess)
#         if abs(guess) > 0.13:
#             score += guess
#             scoringPara += 1
#     score /= scoringPara
#     print(score)
#     print(i)
#     bigModel = Model.predict(artCont, topic)
#     # print(bigModel)
#     lstOfStuff.append(set(i, score, bigModel))
# print(lstOfStuff)
# exit()
# results = Model.predict("I absolutely hate cats", "trump")
# print(results["totalScore"])
# print(f"positive: {results['posScore']}")
# print(f'negative: {results["negScore"]}')
