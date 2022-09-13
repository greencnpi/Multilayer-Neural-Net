import numpy as np, nltk.tokenize, re, ast, math
from numpy import dot
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
numberizer = CountVectorizer(min_df=0, lowercase=False)
lemmatizer = WordNetLemmatizer()
def sigmoid(x):
  return 1/(1+(math.e**-x))
def dsigmoid(x):
  return sigmoid(x)*(sigmoid(1-x))
file = "data.txt"
with open(file) as f:
  content = f.readlines()
content = [x.strip() for x in content]
target = []
messages_start = []
for i in content:
  i = i.split()
  targetpiece = i[-1]
  target.append(ast.literal_eval(targetpiece))
  messages_start.append(" ".join(i[:-1]))

#Message processing
for i in messages_start:
  i = re.sub(r'[^ a-z A-Z 0-9]', " ", i)
  i = nltk.tokenize.word_tokenize(i)
  i = " ".join([lemmatizer.lemmatize(j.lower()) for j in i])
numberizer.fit(messages_start)
numberizer.vocabulary_

#Vars
start = numberizer.transform(messages_start).toarray()
hn_amount = len(start[0])
hnodes = [hn_amount,hn_amount]
speed = 0.2
epochs = 30000

#var processing
seed = 1
start = np.array(start)
target = np.array(target)

#Generated vars
np.random.seed(seed)
layers = 2+len(hnodes)
nodelist = []
nodelist.append(len(start[0]))
for i in hnodes:
  nodelist.append(i)
nodelist.append(np.shape(target)[1])
masternet = []
for i in range(layers-1):
  masternet.append([np.random.rand(nodelist[i],nodelist[i+1]),np.random.rand(1,nodelist[i+1])])

while 1:
  text = input("Input: ")

  #text processing
  textlist = text.split()
  textisbinary = True
  for i in textlist:
    if i == "1" or i == "0":
      pass
    else:
      textisbinary = False

  #Training
  if text == "train":
    miniepoch = round(epochs/20)
    for e in range(20):
      for epoch in range(miniepoch):
        masterlayer = [start]
        for i in range(layers-1):
          masterlayer.append(sigmoid(dot(masterlayer[i], masternet[i][0]) + masternet[i][1]))

        #first change
        error = masterlayer[-1] - target
        output_change = error*dsigmoid(masterlayer[-1])
        adjustments = [speed*dot(masterlayer[-2].T, output_change)]
        for i in output_change:
          masternet[-1][1] -= speed*i
      
        #hidden changes
        for i in range(layers-2):
          error = dot(output_change, masternet[-(i+1)][0].T)
          output_change = error*dsigmoid(masterlayer[-(i+2)])
          adjustments.append(speed*dot(masterlayer[-(i+3)].T, output_change))
          for j in output_change:
            masternet[-1*(i+2)][1] -= speed*j
      
        #adjust weights
        for i,j in zip(adjustments[::-1],masternet):
          j[0] -= i
      print(''.join((str((e+1)*5),"% complete")))
      
    print("done\n")
    continue
  
  elif text == "net":
    counter = 0
    for i in masternet:
      print(str(counter)+"w: \n" + str(i[0]))
      print(str(counter)+"b: \n" + str(i[1]))
      counter += 1
    print("")
    continue

  else:
    #Test, prepare message
    i = re.sub(r'[^ a-z A-Z 0-9]', " ", text)
    i = nltk.tokenize.word_tokenize(i)
    i = [" ".join([lemmatizer.lemmatize(j.lower()) for j in i])]
    numberizer.fit(messages_start)
    numberizer.vocabulary_

    masterlayer = []
    masterlayer.append(numberizer.transform(i).toarray())
    for i in range(layers-1):
      masterlayer.append(sigmoid(np.dot(masterlayer[i], masternet[i][0]) + masternet[i][1]))
    print(str(["{:0.5f}".format(x) for x in np.round(masterlayer[-1][0],6)])+"\n")


