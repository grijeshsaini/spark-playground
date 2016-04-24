package com.spark.playground

import org.apache.spark.{SparkContext, SparkConf}
import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}
import org.apache.spark.mllib.linalg.Vectors

/**
  * Created by grijesh on 23/4/16.
  */

object SparkClustering extends App {

  val sparkConf = new SparkConf()
    .setAppName("com.spark.playground.SparkClustering")
    .setMaster("local")

  val sc = new SparkContext(sparkConf)

  // Load and parse the data
  val data = sc.textFile("/home/grijesh/sampleData/k-means-data.txt")
  val parsedData = data.map(s => Vectors.dense(s.split(' ').map(_.toDouble))).cache()

  // Cluster the data into two classes using KMeans
  val numClusters = 2
  val numIterations = 20
  val clusters = KMeans.train(parsedData, numClusters, numIterations)

  // Evaluate clustering by computing Within Set Sum of Squared Errors
  val WSSSE = clusters.computeCost(parsedData)
  println("Within Set Sum of Squared Errors = " + WSSSE)

  // Save and load model
  clusters.save(sc, "/home/grijesh/myModelPath")
  val sameModel = KMeansModel.load(sc, "/home/grijesh/myModelPath")
}