from bs4 import BeautifulSoup
import re
import nltk
import os
import numpy as np
from collections import Counter #集合
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from  nltk.corpus import stopwords

def mail_text_preprocessing(mail, remove_stopwords):
    # 去掉html标记
    raw_text = BeautifulSoup(mail, 'html').get_text()
    # 去掉非字母字符
    letters = re.sub('[^a-zA-Z-]', ' ', raw_text)
    words = letters.lower().split()  # 将英文词汇转换为小写形式
    # 清除停用词
    if remove_stopwords:
        stop_words = set(stopwords.words('english'))
        # 初始化stemmer寻找各个词汇的最原始的词根
        # stemmer=nltk.stem.PorterStemmer()
        # words=[stemmer.stem(w) for w in words if w not in stop_words]
        words = [w for w in words if w not in stop_words]
        # 处理上面保留短'-'引发的问题,去掉单独的短'-'数据。
        result = []
        for item in words:
            if item == '-':
                continue
            elif len(item) == 1:
                # 过滤掉单个字母形式的词汇
                continue
            else:
                result.append(item)
    return result



train_path = 'D:/shujuji/train-mails/'


# 构造词典(词表)
def make_Dictionary(train_dir):
    emails = [os.path.join(train_dir, f) for f in os.listdir(train_dir)]
    all_words = []
    for mail in emails:
        with open(mail,encoding='gb18030',errors='ignore') as m:
            for i, line in enumerate(m):
                if i == 0:  # body of email is only 3rd line of text file.
                    # words=line.split()
                    words = mail_text_preprocessing(line, True)  # 在这里对文档的每一行进行NLP语言文字的数据预处理
                    all_words += words
    dictionary = Counter(all_words)  # 计数函数,直接对一个句子中的各个单词进行计数
    # 删掉了一些与垃圾邮件的判定无关的单字符==那些非文字类符号
    list_to_remove = list(dictionary.keys())  # 需要转为list来存储,不然原始的代码这里有问题。
    jinlist = ['com', 'edu', 'www', 'ffa', 'etc']
    jinli = [w for w in list_to_remove if len(w) == 2]
    paichu = ['us', 'he', 'ad', 'me', 'cd', 'id', 'ps', 'pi']


    jinlist += jinli
    for i in range(len(paichu)):
       try:
          del jinlist[jinlist.index(paichu[i])]  # paichu列表中的不删除
       except:
          continue
    for item in list_to_remove:
       if item.isalpha() == False:  # 不是字符的删除,如数字和特殊符号均删除,只保留26个字母的字符串。
           del dictionary[item]
       elif len(item) == 1:  # 单个字符删除
           if item != 'I':
              del dictionary[item]
           else:
              pass
       elif item in jinlist:
           del dictionary[item]
    dictionary = dictionary.most_common(300)
    return dictionary
# 通过输入 print dictionary 指令就可以输出词典==词典里应该会有以下这些高频词（本例中我们选取了频率最高的 3000 个词）
dictionary = make_Dictionary(train_path)


# 特征提取
def extract_features(mail_dir):
    # mail_dir=train_path
    files = [os.path.join(mail_dir, fi) for fi in os.listdir(mail_dir)]
    features_matrix = np.zeros((len(files), 300))  # 这里的维度大小使用()传入。
    docID = 0
    for fil in files:
        with open(fil,encoding='gb18030',errors='ignore') as fi:
            for i, line in enumerate(fi):
                if i == 0:
                    # words=line.split()
                    words = mail_text_preprocessing(line, True)  # 在这里对文档的每一行进行NLP语言文字的数据预处理
                    for word in words:
                        wordID = 0
                        for i, d in enumerate(dictionary):
                            if d[0] == word:
                                wordID = i
                                """构造邮件的特征行的方法"""
                                # 使用词频填充每封邮件的特征行,即表征一封邮件样本
                                features_matrix[docID, wordID] += words.count(word)
                                # 使用0或1填充每封邮件的特征行,表征对应词汇是否出现在邮件中,即表征一封邮件样本
                                # features_matrix[docID,wordID]=1

        docID = docID + 1
    return features_matrix


train_labels = np.zeros(1552)
train_labels[776:1551] = 1
train_matrix = extract_features(train_path)

# 训练模型


model1 = LinearSVC()
model2 = MultinomialNB()
model3 = DecisionTreeClassifier()

model1.fit(train_matrix, train_labels)
model2.fit(train_matrix, train_labels)
model3.fit(train_matrix,train_labels)


# 测试训练好的模型对垃圾评论的分类预测情况
test_dir = 'D:/shujuji/test-mails/'
test_matrix = extract_features(test_dir)
test_labels = np.zeros(350)
test_labels[175:350] = 1

result1 = model1.predict(test_matrix)
result2 = model2.predict(test_matrix)
result3 = model3.predict(test_matrix)

# 分类预测结果评估
print(confusion_matrix(test_labels, result1))
print(confusion_matrix(test_labels, result2))
print(confusion_matrix(test_labels, result3))




