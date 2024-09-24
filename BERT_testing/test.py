from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Load the ABSA model and tokenizer
model_name = "yangheng/deberta-v3-base-absa-v1.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

for aspect in ['camera', 'phone']:
   print(aspect, classifier('The camera quality of this phone is amazing.',  text_pair=aspect))
   