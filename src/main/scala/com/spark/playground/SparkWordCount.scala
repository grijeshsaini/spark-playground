package com.spark.playground

import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by grijesh on 23/4/16.
  */
object SparkWordCount extends App{
  val sparkConf = new SparkConf()
    .setAppName("com.spark.playground.SparkWordCount")
    .setMaster("local")

  val sc = new SparkContext(sparkConf)

  /*creating an inputRDD to read text file (in.txt) through Spark context*/
  val input = sc.textFile("/home/grijesh/nounlist.txt")

  /* Transform the inputRDD into countRDD */
  val count = input.flatMap(line ⇒ line.split(" "))
    .map(word ⇒ (word, 1))
    .reduceByKey(_ + _)

  /* saveAsTextFile method is an action that effects on the RDD */
  count.saveAsTextFile("/home/grijesh/outfile")
  println("Done")
}
