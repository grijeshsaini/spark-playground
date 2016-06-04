package com.spark.playground

import org.apache.spark.ml.clustering.{KMeansModel, KMeans}
import org.apache.spark.ml.feature.{IDF, HashingTF, Tokenizer}
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by grijesh on 4/6/16.
  */
object MovieClustering extends App {

  val sparkConf = new SparkConf()
  .setAppName("com.spark.playground.MovieClustering")
  .setMaster("local")
  val sc = new SparkContext(sparkConf)
  val sqlContext = new org.apache.spark.sql.SQLContext(sc)
  import sqlContext.implicits._
  val documents = sc.textFile("/home/grijesh/sampleData/text-data.txt").map(_.split(" ")).map(movie => Movie(movie(0), movie(1))).toDF()

  //documents.registerTempTable("movies")

  val tokenizer = new Tokenizer().setInputCol("movieType").setOutputCol("words")
  val wordsData = tokenizer.transform(documents)
  val hashingTF = new HashingTF().setInputCol("words").setOutputCol("rawFeatures")
  val featurizedData = hashingTF.transform(wordsData)
  val idf = new IDF().setInputCol("rawFeatures").setOutputCol("features")
  val idfModel = idf.fit(featurizedData)
  val tfIdf = idfModel.transform(featurizedData)
  //val tfIdfDetails = tfIdf.select("features", "name")


  val kmeans = new KMeans().setK(3).setFeaturesCol("features").setPredictionCol("prediction")
  val model:KMeansModel = kmeans.fit(tfIdf)

  val output = model.transform(tfIdf)
  output.show()
  //val output = tfIdfDetails.map(details=> (details(1),model.predict(details(0).asInstanceOf[org.apache.spark.mllib.linalg.Vector])))
  /*val numClusters = 3
  val numIterations = 20
  val clusters = KMeans.train(tfidf, numClusters, numIterations)*/

  case class Movie(name: String, movieType: String)


}
