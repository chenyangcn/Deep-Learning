from sklearn.feature_extraction import DictVextorizer
import csv
from sklearn import preprocessing
from sklearn import tree
from sklearn.externals.six import StringIO

file = open(r'./AllElectronics.csv', 'rb')
reader = csv.reader(file)
headers = reader.next()

print(headers)
