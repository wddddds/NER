package uk.ac.ucl.cs.mr.statnlpbook.assignment2

import uk.ac.ucl.cs.mr.statnlpbook.assignment2

import scala.collection.mutable

/**
 * Created by Georgios on 30/10/2015.
 */

object Problem5{

  def main (args: Array[String]) {
    println("Joint Extraction")

    val train_dir = "./data/assignment2/bionlp/train"
    val test_dir = "./data/assignment2/bionlp/test"

    // load train and dev data
    // read the specification of the method to load more/less data for debugging speedup
    val (trainDocs, devDocs) = BioNLP.getTrainDevDocuments(train_dir,0.8,500)
    // val (trainDocs, devDocs) = BioNLP.getTrainDevDocuments(train_dir,0.8,100)
    // load test
    val testDocs = BioNLP.getTestDocuments(test_dir)
    // make tuples (Candidate,Gold)
    def preprocess(candidates: Seq[Candidate]) = candidates.map(e => e -> (e.gold,e.arguments.map(_.gold)))

    // ================= Joint Classification =================
    // get candidates and make tuples with gold
    // read the specifications of the method for different subsampling thresholds
    // no subsampling for dev/test!
    // def getJointCandidates(docs: Seq[Document]) = docs.flatMap(_.jointCandidates(0.02,0.4))
    def getJointCandidates(docs: Seq[Document]) = docs.flatMap(_.jointCandidates(0.8,0.4))
    def getTestJointCandidates(docs: Seq[Document]) = docs.flatMap(_.jointCandidates())
    val jointTrain = preprocess(getJointCandidates(trainDocs))
    val jointDev = preprocess(getTestJointCandidates(devDocs))
    val jointTest = preprocess(getTestJointCandidates(testDocs))

    // show statistics for counts of true labels, useful for deciding on subsampling
    println("True label counts (trigger - train):")
    println(jointTrain.unzip._2.unzip._1.groupBy(x=>x).mapValues(_.length))
    println("True label counts (trigger - dev):")
    println(jointDev.unzip._2.unzip._1.groupBy(x=>x).mapValues(_.length))
    println("True label counts (argument - train):")
    println(jointTrain.unzip._2.unzip._2.flatten.groupBy(x=>x).mapValues(_.length))
    println("True label counts (argument - dev):")
    println(jointDev.unzip._2.unzip._2.flatten.groupBy(x=>x).mapValues(_.length))


    // get label sets
    val triggerLabels = jointTrain.map(_._2._1).toSet
    val argumentLabels = jointTrain.flatMap(_._2._2).toSet

    // define model
    //TODO: change the features function to explore different types of features
    //TODO: experiment with the unconstrained and constrained (you need to implement the inner search) models
    // val jointModel = JointUnconstrainedClassifier(triggerLabels,argumentLabels,Features.myTriggerFeatures,Features.defaultArgumentFeatures)
    val jointModel = JointConstrainedClassifier(triggerLabels,argumentLabels,Features.myTriggerFeatures,Features.myArgumentFeatures)

    // use training algorithm to get weights of model
    val jointWeights = PrecompiledTrainers.trainPerceptron(jointTrain,jointModel.feat,jointModel.predict,20)

    // get predictions on dev
    val jointDevPred = jointDev.unzip._1.map { case e => jointModel.predict(e,jointWeights) }
    val jointDevGold = jointDev.unzip._2

    // Triggers (dev)
    val triggerDevPred = jointDevPred.unzip._1
    val triggerDevGold = jointDevGold.unzip._1
    val triggerDevEval = Evaluation(triggerDevGold,triggerDevPred,Set("None"))
    println("Evaluation for trigger classification:")
    println(triggerDevEval.toString)

    ErrorAnalysis(jointDev.unzip._1,triggerDevGold,triggerDevPred).showErrors(50)

    // Arguments (dev)
    val argumentDevPred = jointDevPred.unzip._2.flatten
    val argumentDevGold = jointDevGold.unzip._2.flatten
    val argumentDevEval = Evaluation(argumentDevGold,argumentDevPred,Set("None"))
    println("Evaluation for argument classification:")
    println(argumentDevEval.toString)

    // get predictions on test
    val jointTestPred = jointTest.unzip._1.map { case e => jointModel.predict(e,jointWeights) }
    // Triggers (test)
    val triggerTestPred = jointTestPred.unzip._1
    // write to file
    Evaluation.toFile(triggerTestPred,"./data/assignment2/out/joint_trigger_test.txt")
    // Arguments (test)
    val argumentTestPred = jointTestPred.unzip._2.flatten
    // write to file
    Evaluation.toFile(argumentTestPred,"./data/assignment2/out/joint_argument_test.txt")
  }

}

/**
 * A joint event classifier (both triggers and arguments).
 * It predicts the structured event.
 * It's predict method should only produce the best solution that respects the constraints on the event structure.
 * @param triggerLabels
 * @param argumentLabels
 * @param triggerFeature
 * @param argumentFeature
 */
case class JointConstrainedClassifier(triggerLabels:Set[Label],
                                      argumentLabels:Set[Label],
                                      triggerFeature:(Candidate,Label)=>FeatureVector,
                                      argumentFeature:(Candidate,Label)=>FeatureVector
                                       ) extends JointModel {
  /**
   * Constraint 1: if e=None, all a=None
   * Constraint 2: if e!=None, at least one a=Theme
   * Constraint 3: only e=Regulation can have a=Cause
   * @param x
   * @param weights
   * @return
   */


  def predict(x: Candidate, weights: Weights) = {
    def argmax_cached(labels: Set[Label], x: Candidate, weights: Weights, feat:(Candidate,Label)=>FeatureVector) = {
      val scores = labels.toSeq.map(y => y -> dot(feat(x, y), weights)).toMap withDefaultValue 0.0
      (scores.maxBy(_._2), ("Theme", scores("Theme")))
    }

    def best_seq(trigger:Label, arg_set:Set[Label]): (Double, StructuredLabels) ={
      val trigger_score = dot(triggerFeature(x, trigger), weights)
      val arg_score_cache = scala.collection.mutable.ArrayBuffer.empty[((String,Double), (String,Double))]
      for (arg <- x.arguments)
        arg_score_cache += argmax_cached(arg_set,arg,weights,argumentFeature)
      val contains_theme = arg_score_cache.exists(x => x._1._1 == "Theme")
      if (contains_theme){
        // Theme is already there
        val total_score = trigger_score + arg_score_cache.map(_._1._2).sum
        val seq_pred = (trigger, arg_score_cache.map(_._1._1))
        (total_score, seq_pred)
      } else {
        // find minimum gap
        val min_gap_ind = arg_score_cache.map(x => x._1._2 - x._2._2).zipWithIndex.minBy(_._1)._2
        arg_score_cache(min_gap_ind) = (arg_score_cache(min_gap_ind)._2, arg_score_cache(min_gap_ind)._1)
        val total_score = trigger_score + arg_score_cache.map(_._1._2).sum
        val seq_pred = (trigger, arg_score_cache.map(_._1._1))
        (total_score, seq_pred)
      }
    }
    // handle None
    val score_pred = scala.collection.mutable.ArrayBuffer.empty[(Double, StructuredLabels)]
    val none_score = dot(triggerFeature(x, "None"), weights) + x.arguments.map(arg => dot(argumentFeature(arg, "None"), weights)).sum
    val none_ret = ("None", for (arg<-x.arguments) yield "None")
    score_pred += (none_score -> none_ret)
    // handle regulation
    val reg_lab = List("Positive_regulation", "Negative_regulation", "Regulation")
    for (tri <- reg_lab) score_pred += best_seq(tri, argumentLabels)
//    val reg_tup = best_seq("Regulation", argumentLabels)
//    score_pred += reg_tup
    // handle others
    val arg_label_no_cause = argumentLabels - "Cause"
    val tri_label_rest = triggerLabels - ("None", "Regulation", "Positive_regulation", "Negative_regulation")
    for (tri <- tri_label_rest) score_pred += best_seq(tri, arg_label_no_cause)
    // find the best prediction
    score_pred.maxBy(_._1)._2


  }


}


/**
 * A joint event classifier (both triggers and arguments).
 * It predicts the structured event.
 * It treats triggers and arguments independently, i.e. it ignores any solution constraints.
 * @param triggerLabels
 * @param argumentLabels
 * @param triggerFeature
 * @param argumentFeature
 */
case class JointUnconstrainedClassifier(triggerLabels:Set[Label],
                                        argumentLabels:Set[Label],
                                        triggerFeature:(Candidate,Label)=>FeatureVector,
                                        argumentFeature:(Candidate,Label)=>FeatureVector
                                         ) extends JointModel{
  def predict(x: Candidate, weights: Weights) = {
    def argmax(labels: Set[Label], x: Candidate, weights: Weights, feat:(Candidate,Label)=>FeatureVector) = {
      val scores = labels.toSeq.map(y => y -> dot(feat(x, y), weights)).toMap withDefaultValue 0.0
      scores.maxBy(_._2)._1
    }
    val bestTrigger = argmax(triggerLabels,x,weights,triggerFeature)
    val bestArguments = for (arg<-x.arguments) yield argmax(argumentLabels,arg,weights,argumentFeature)
    (bestTrigger,bestArguments)
  }

}

trait JointModel extends Model[Candidate,StructuredLabels]{
  def triggerFeature:(Candidate,Label)=>FeatureVector
  def argumentFeature:(Candidate,Label)=>FeatureVector
  def feat(x: Candidate, y: StructuredLabels): FeatureVector ={
    val f = new mutable.HashMap[FeatureKey, Double] withDefaultValue 0.0
    addInPlace(triggerFeature(x,y._1),f,1)
    for ((a,label)<- x.arguments zip y._2){
      addInPlace(argumentFeature(a,label),f,1)
    }
    f
  }
}


