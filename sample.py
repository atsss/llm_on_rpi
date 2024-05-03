from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import TFAutoModelForSequenceClassification
import tflite_runtime as tf

model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")


tf_model = TFAutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english", from_pt=True)
tf_model.save_pretrained("./distilbert_tf")
converter = tf.lite.TFLiteConverter.from_saved_model("./distilbert_tf")
tflite_model = converter.convert()
open("distilbert.tflite", "wb").write(tflite_model)


interpreter = tf.lite.Interpreter(model_path="distilbert.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

text = "This movie is awesome!"
inputs = tokenizer(text, return_tensors="tf")
input_ids = inputs["input_ids"].numpy()
attention_mask = inputs["attention_mask"].numpy()

interpreter.set_tensor(input_details[0]["index"], input_ids)
interpreter.set_tensor(input_details[1]["index"], attention_mask)
interpreter.invoke()

output = interpreter.get_tensor(output_details[0]["index"])
prediction = tf.argmax(output, axis=1).numpy()[0]
label = ["negative", "positive"][prediction]
print(f"Text: {text}")
print(f"Label: {label}")
