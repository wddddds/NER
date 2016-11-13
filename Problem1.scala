package uk.ac.ucl.cs.mr.statnlpbook.assignment2

import scala.collection.mutable

/**
 * Created by Georgios on 11/11/2015.
 */
object Problem1 {


  /**
   * Train a linear model using the perceptron algorithm.
   * @param instances the training instances.
   * @param feat a joint feature function.
   * @param predict a prediction function that maps inputs to outputs using the given weights.
   * @param iterations number of iterations.
   * @param learningRate
   * @tparam X type of input.
   * @tparam Y type of output.
   * @return a linear model trained using the perceptron algorithm.
   */
  def trainPerceptron[X, Y](instances: Seq[(X, Y)],
                            feat: (X, Y) => FeatureVector,
                            predict: (X, Weights) => Y,
                            iterations: Int = 2,
                            learningRate:Double = 1.0): Weights = {
    val mutableWeights = scala.collection.mutable.Map[FeatureKey, Double]().withDefaultValue(0)
    def instance_iterator(instances: Seq[(X, Y)], iterations:Int): Iterator[(X, Y)] = new Iterator[(X, Y)]{
      val total_instance = instances.length * iterations
      var current_instance = 0
      def hasNext = current_instance <= total_instance
      def next():(X, Y) = {
        val seq_ind = current_instance % instances.length
        current_instance = current_instance + 1
        instances(seq_ind)
      }
    }
    val complete_instances = instance_iterator(instances, iterations)

    complete_instances.foreach( x => {
      val c_predict = predict(x._1, mutableWeights)
      if (c_predict != x._2){
        val c_feat = feat(x._1, x._2)
        val w_feat = feat(x._1, c_predict)
        addInPlace(c_feat, mutableWeights, learningRate)
        addInPlace(w_feat mapValues (x => -x), mutableWeights, learningRate)
      }
    })
    mutableWeights.toMap.withDefaultValue(0)

  }

  def trainMaximumEnt[X, Y](instances: Seq[(X, Y)],
                            labels:Set[Y],
                            feat: (X, Y) => FeatureVector,
                            iterations: Int = 2,
                            learningRate:Double = 1.0): Weights = {
    def avgFeat(feats: Seq[FeatureVector], feat_prob: Seq[Double]): FeatureVector = {
      val mutableFeats = scala.collection.mutable.Map[FeatureKey, Double]().withDefaultValue(0)
      feats.zip(feat_prob).foreach( x =>{
        for ((k,v) <- x._1) mutableFeats(k) += v * x._2
      })
      mutableFeats.toMap.withDefaultValue(0)
    }

    def Gradient(instance: (X, Y), w:Weights): FeatureVector = {
      // get expected feature
      val xfeat = feat(instance._1:X, _:Y)
      val feats = labels.toSeq.map(xfeat)
      val wdot_exp: (FeatureVector => Double) = x => math.exp(dot(x, w))
      val scores = feats.map(wdot_exp)
      val sum_of_scores = scores.sum
      val feat_prob = scores.map(x => x/sum_of_scores)
      val expected_feat = avgFeat(feats, feat_prob)
      // get best feature

//      val best_ind = scores.zipWithIndex.maxBy(_._1)._2
      val best_feat = feat(instance._1:X, instance._2:Y)
      // compute gradient
      val grad = scala.collection.mutable.Map[FeatureKey, Double]().withDefaultValue(0)
      for ((k, v) <- expected_feat) grad(k) += (best_feat.getOrElse(k, 0.0) - v)
      grad.toMap.withDefaultValue(0)
    }

    def instance_iterator(instances: Seq[(X, Y)], iterations:Int): Iterator[(X, Y)] = new Iterator[(X, Y)]{
      val total_instance = instances.length * iterations
      var current_instance = 0
      def hasNext = current_instance <= total_instance
      def next():(X, Y) = {
        val seq_ind = current_instance % instances.length
        current_instance = current_instance + 1
        instances(seq_ind)
      }
    }

    val mutableWeights = scala.collection.mutable.Map[FeatureKey, Double]().withDefaultValue(0)
    val complete_instances = instance_iterator(instances, iterations)
    complete_instances.foreach( x => {
      addInPlace(Gradient(x, mutableWeights) ,mutableWeights, learningRate)
    })
    mutableWeights.toMap.withDefaultValue(0)
  }
  /**
   * Run this code to evaluate your implementation of your perceptron algorithm trainer
   * Results should be similar to the precompiled trainer
   * @param args
   */
  def main (args: Array[String] ) {

    val train_dir = "./data/assignment2/bionlp/train"

    // load train and dev data
    // read the specification of the method to load more/less data for debugging speedup
    val (trainDocs, devDocs) = BioNLP.getTrainDevDocuments(train_dir, 0.8, 500)
    // make tuples (Candidate,Gold)
    def preprocess(candidates: Seq[Candidate]) = candidates.map(e => e -> e.gold)

    // ================= Trigger Classification =================

    // get candidates and make tuples with gold
    // read the specifications of the method for different subsampling thresholds
    // no subsampling for dev/test!
    def getTriggerCandidates(docs: Seq[Document]) = docs.flatMap(_.triggerCandidates(0.02))
    def getTestTriggerCandidates(docs: Seq[Document]) = docs.flatMap(_.triggerCandidates())
    val triggerTrain = preprocess(getTriggerCandidates(trainDocs))
    val triggerDev = preprocess(getTestTriggerCandidates(devDocs))

    // get label set
    val triggerLabels = triggerTrain.map(_._2).toSet

    // define model
    val triggerModel = SimpleClassifier(triggerLabels, defaultTriggerFeatures)

    val myWeights = trainPerceptron(triggerTrain, triggerModel.feat, triggerModel.predict, 1)
    val precompiledWeights = PrecompiledTrainers.trainPerceptron(triggerTrain, triggerModel.feat, triggerModel.predict, 1)
//    val entWeights = trainMaximumEnt(triggerTrain, triggerLabels, triggerModel.feat, 20, 0.15)
    // get predictions on dev
    val (myPred, gold) = triggerDev.map { case (trigger, gold) => (triggerModel.predict(trigger, myWeights), gold) }.unzip
//    val (myEnt, _) = triggerDev.map { case (trigger, gold) => (triggerModel.predict(trigger, entWeights), gold) }.unzip
    val (precompiledPred, _) = triggerDev.map { case (trigger, gold) => (triggerModel.predict(trigger, precompiledWeights), gold) }.unzip

    // evaluate models (dev)
    println(" ================= Trigger Evaluation ================= ")
    println("Evaluation - my trainer:")
    println(Evaluation(gold, myPred, Set("None")).toString)
    println("Evaluation - precompiled trainer:")
    println(Evaluation(gold, precompiledPred, Set("None")).toString)
//    println("Evaluation - MaxEnt trainer:")
//    println(Evaluation(gold, myEnt, Set("None")).toString)


    // ================= Argument Classification =================

    // get candidates and make tuples with gold
    // no subsampling for dev/test!
    def getArgumentCandidates(docs:Seq[Document]) = docs.flatMap(_.argumentCandidates(0.008))
    def getTestArgumentCandidates(docs:Seq[Document]) = docs.flatMap(_.argumentCandidates())
    val argumentTrain =  preprocess(getArgumentCandidates(trainDocs))
    val argumentDev = preprocess(getTestArgumentCandidates(devDocs))

    // get label set
    val argumentLabels = argumentTrain.map(_._2).toSet

    // define model
    val argumentModel = SimpleClassifier(argumentLabels, defaultArgumentFeatures)

    //val argumentWeights = PrecompiledTrainers.trainNB(argumentTrain,argumentModel.feat)
    val myargWeights = trainPerceptron(argumentTrain,argumentModel.feat,argumentModel.predict,1)
    val precompiledargWeights = PrecompiledTrainers.trainPerceptron(argumentTrain,argumentModel.feat,argumentModel.predict,1)

//    val entargWeights = trainMaximumEnt(argumentTrain, argumentLabels, argumentModel.feat, 20, 0.15)

    // get predictions on dev
    val (argumentDevPredmy, argumentDevGold) = argumentDev.map { case (arg, gold) => (argumentModel.predict(arg,myargWeights), gold) }.unzip
    val (argumentDevPredpre, _) = argumentDev.map { case (arg, gold) => (argumentModel.predict(arg,precompiledargWeights), gold) }.unzip
//    val (argumentDevPredent, _) = argumentDev.map { case (arg, gold) => (argumentModel.predict(arg,entargWeights), gold) }.unzip
    // evaluate on dev
    // evaluate models (dev)
    println(" ================= Argument Evaluation ================= ")
    println("Evaluation - my trainer:")
    println(Evaluation(argumentDevGold, argumentDevPredmy, Set("None")).toString)
    println("Evaluation - precompiled trainer:")
    println(Evaluation(argumentDevGold, argumentDevPredpre, Set("None")).toString)
//    println("Evaluation - MaxEnt trainer:")
//    println(Evaluation(argumentDevGold, argumentDevPredent, Set("None")).toString)


  }

  def defaultTriggerFeatures(x: Candidate, y: Label): FeatureVector = {
    val doc = x.doc
    val begin = x.begin
    val end = x.end
    val thisSentence = doc.sentences(x.sentenceIndex) //use this to gain access to the parent sentence
    val feats = new mutable.HashMap[FeatureKey,Double]
    feats += FeatureKey("label bias", List(y)) -> 1.0 //bias feature
    val token = thisSentence.tokens(begin) //first token of Trigger
    feats += FeatureKey("first trigger word", List(token.word, y)) -> 1.0 //word feature
    feats.toMap
  }

  def defaultArgumentFeatures(x: Candidate, y: Label): FeatureVector = {
    val doc = x.doc
    val begin = x.begin
    val thisSentence = doc.sentences(x.sentenceIndex)
    val event = thisSentence.events(x.parentIndex) //use this to gain access to the parent event
    val feats = new mutable.HashMap[FeatureKey,Double]
    feats += FeatureKey("argument label bias", List(y)) -> 1.0
    val token = thisSentence.tokens(begin) //first word of argument
    feats += FeatureKey("first argument word", List(token.word, y)) -> 1.0
    feats.toMap
  }


}
