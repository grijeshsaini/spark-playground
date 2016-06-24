package com.spark.playground

import org.apache.spark.ml.clustering.{KMeans, KMeansModel}
import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer}
import org.apache.spark.{SparkConf, SparkContext}

import scala.language.postfixOps

/**
  * Created by grijesh on 4/6/16.
  */
object GenericClustering extends App {

  val sparkConf = new SparkConf()
    .setAppName("com.spark.playground.GenericClustering")
    .setMaster("local")
  val sc = new SparkContext(sparkConf)
  val sqlContext = new org.apache.spark.sql.SQLContext(sc)

  import sqlContext.implicits._

  //args foreach println
  if (args.length < 3) {
    sc.stop()
    throw new IllegalArgumentException("3 Arguments required in order of InputPath Separator keyColumn")
  }

  val inputArguments = InputArguments(args(0), args(1), args(2))

  val documents = sc.textFile(inputArguments.inputPath)
    .map(_.split(inputArguments.separator))
    .map(data => ParsedData(data(inputArguments.keyColumn.toInt), data.mkString(inputArguments.separator)))
    .toDF()

  val tokenizer = new Tokenizer().setInputCol("keyColumn").setOutputCol("tokenizedData")
  val tokenizedData = tokenizer.transform(documents)
  val hashingTF = new HashingTF().setInputCol("tokenizedData").setOutputCol("rawFeatures")
  val featurizedData = hashingTF.transform(tokenizedData)
  val idf = new IDF().setInputCol("rawFeatures").setOutputCol("features")
  val idfModel = idf.fit(featurizedData)
  val tfIdf = idfModel.transform(featurizedData)
  val kmeans = new KMeans().setK(3).setFeaturesCol("features").setPredictionCol("prediction").setMaxIter(20)
  val model: KMeansModel = kmeans.fit(tfIdf)

  val output = model.transform(tfIdf)

  output.show()

  sc.stop()
}

case class ParsedData(keyColumn: String, data: String)

//TODO Escape Separator as an ex: //| (Can use StringEsacpeUtils from apache commons)
//TODO need to decide where this escaping needs to be done in algorithm or in invoker
case class InputArguments(inputPath: String, separator: String, keyColumn: String) {
  def apply(inputPath: String, separator: String, keyColumn: String): InputArguments = {
    require(inputPath == null, "Input Path should not be null")
    require(separator == null, "Separator should not be null")
    require(keyColumn == null, "KeyColumn should not be null")
    new InputArguments(inputPath, separator, keyColumn)
  }

  /*require(keyColumn match {
    case _:Int => false
    case _:_ => true
  }, "KeyColumn should be Integer")*/
}
