package com.spark.playground

import org.apache.spark.mllib.clustering.{KMeansModel, KMeans}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.{SparkContext, SparkConf}
import scalax.chart._

/**
  * Created by grijesh on 24/4/16.
  */
object KmeansOnIris extends App{

  val sparkConf = new SparkConf()
    .setAppName("com.spark.playground.KmeansOnIris")
    .setMaster("local")

  val sc = new SparkContext(sparkConf)

  // Load and parse the data
  val data = sc.textFile("/home/grijesh/sampleData/iris-data.txt")

  val parsedData = data.map(s => Vectors.dense(s.split(',').dropRight(1).map(_.toDouble))).cache()

  // Cluster the data into two classes using KMeans
  val numClusters = 3
  val numIterations = 20
  val clusters = KMeans.train(parsedData, numClusters, numIterations)

  // Evaluate clustering by computing Within Set Sum of Squared Errors
  val WSSSE = clusters.computeCost(parsedData)
  println("Within Set Sum of Squared Errors = " + WSSSE)

  //clusters.save(sc, "/home/grijesh/irisOutput")

  clusters.toPMML(System.out)

  val petalDetails = parsedData.map(arr => (clusters.predict(arr),arr.toArray.drop(2).toVector))

  val sepalDetails = parsedData.map(arr => (clusters.predict(arr),arr.toArray.dropRight(2).toVector))

}
