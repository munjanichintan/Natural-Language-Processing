# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 20:20:38 2020

@author: Chintan Munjani
"""

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# This is a speech of Elon Musk in interview.

paragraph = """If you have a vision or a dream and there is no well-trodden path to it,
               you can’t just give up. You have to take the risk and follow your heart even when you have to go all alone. 
               A few years ago, nobody would have thought that electric car could be used as your main car. 
               But Elon Musk took the risk and built a mainstream electric car with Tesla Motors.
               The market rewards risk as much as it rewards knowledge. If you’re taking the risk, 
               you’re doing something new and unique, and since few competitors would take the same risks, 
               the market will reward you handsomely for it. If Elon Musk hadn’t taken the risk, 
               this wouldn’t be one of the greatest motivational speeches on the web. Risk is rewarded when played well.
               Whether you have a job or a business, you’re producing a product with your work. 
               And to make your product stand out, you need to listen to feedback. 
               You have to take your product and put it in front of knowledgeable people, 
               and even your friends, to provide you with valuable feedback.
               Weak people get discouraged by criticism. Strong people use it to their advantage. 
               They take the feedback from others, see if there is any truth to it, 
               make the improvements in their product if needed and raise their game up a notch. 
               This is how winning is done!"""

# Convert paragraph to sentences.
sentences = nltk.sent_tokenize(paragraph)
lemmatizer = WordNetLemmatizer()

# Lemmatization
for i in range(len(sentences)):
    words = nltk.word_tokenize(sentences[i])
    word = [lemmatizer.lemmatize(word) for word in words if word not in set(stopwords.words('english'))]
    sentences[i] = ' '.join(word)