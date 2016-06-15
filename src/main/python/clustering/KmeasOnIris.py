from pyspark.mllib.clustering import KMeans, KMeansModel
from numpy import array
from math import sqrt
from time import time

# Load and parse the iris-data (https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data)
t0 = time()
data = sc.textFile("/home/grijesh/sampleData/iris-data.txt")
tt = time() - t0
print "Data has loaded successfully in {}".format(round(tt,3))
t0 = time()
parsedData = data.map(lambda line: array([float(x) for x in line.split(',')[:-1]]))
tt = time() - t0
print "{} taken to parse the data with count {}".format(round(tt,3),parsedData.count())
head_rows = parsedData.take(5)
print(head_rows)

# Build the model (cluster the data)
clusters = KMeans.train(parsedData, 3, maxIterations=10,
                        initializationMode="random")

# Evaluate clustering by computing Within Set Sum of Squared Errors
def error(point):
    center = clusters.centers[clusters.predict(point)]
    return sqrt(sum([x**2 for x in (point - center)]))

WSSSE = parsedData.map(lambda point: error(point)).reduce(lambda x, y: x + y)
print "Within Set Sum of Squared Error {}".format(str(WSSSE))

# Can use below API also to evaluate WSSSE
#WSSSE = clusters.computeCost(parsedData)

# Save and load model
#clusters.save(sc, "myModelPath")
#sameModel = KMeansModel.load(sc, "myModelPath")
#totalArray = parsedData.map(lambda line: (clusters.predict(line),line.tolist()))
petalDetails = parsedData.map(lambda line: (clusters.predict(line),line.tolist()[:2]))
sepalDetails = parsedData.map(lambda line: (clusters.predict(line),line.tolist()[2:]))