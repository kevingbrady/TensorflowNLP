import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

if __name__ == '__main__':

    for word in stopwords.words("english"):
        print('\" ' + word + ' \", ', end='')
