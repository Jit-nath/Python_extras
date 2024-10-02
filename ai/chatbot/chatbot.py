import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.preprocessing import LabelEncoder
import random
import json
import nltk

# Download NLTK data
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Initialize lemmatizer to get base form of words
lemmatizer = WordNetLemmatizer()

# Load the dataset (intents)
data = {
    "intents": [
        {"tag": "greeting", "patterns": ["Hello", "Hi", "Hey", "Hi there", "Hello there"], "responses": ["Hello!", "Hi there!", "Hey!"]},
        {"tag": "goodbye", "patterns": ["Bye", "See you later", "Goodbye"], "responses": ["Goodbye!", "See you later!"]},
        {"tag": "thanks", "patterns": ["Thanks", "Thank you", "That's helpful"], "responses": ["You're welcome!", "Happy to help!"]},
        {"tag": "how_are_you", "patterns": ["How are you?", "How's it going?"], "responses": ["I'm good, thanks!", "Doing well! How about you?"]},
        {"tag": "name","patterns": ["what is your name?","Your name?"], "responses":["My name is Omega","Omega"]}
    ]
}

# Prepare data for training
words = []
classes = []
documents = []
ignore_words = ['?', '!']

# Tokenize and lemmatize the words from patterns
for intent in data['intents']:
    for pattern in intent['patterns']:
        word_list = word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(set(words))

classes = sorted(set(classes))

# Prepare training data
training = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = []
    word_patterns = doc[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1 if word in word_patterns else 0)

    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])

# Check for consistency in lengths
print(f"Example bag length: {len(training[0][0])}")
print(f"Example output_row length: {len(training[0][1])}")

# Convert to NumPy arrays
training = np.array(training, dtype=object)
train_x = np.array([item[0] for item in training])
train_y = np.array([item[1] for item in training])

# Check shapes
print("Training data shape (X):", train_x.shape)
print("Training data shape (Y):", train_y.shape)

# Build the model
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)

# Save the model
model.save('chatbot_model.h5')

# Load the trained model
model = tf.keras.models.load_model('chatbot_model.h5')

# Helper functions for predictions
def clean_sentence(sentence):
    sentence_words = word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence, words):
    sentence_words = clean_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence, model):
    bow = bag_of_words(sentence, words)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    for i in intents_json['intents']:
        if i['tag'] == tag:
            return random.choice(i['responses'])

# Running the chatbot
print("Chatbot is running! (type 'quit' to exit)")

while True:
    message = input("You: ")
    if message.lower() == "quit":
        break

    intents = predict_class(message, model)
    response = get_response(intents, data)
    print("Bot:", response)
