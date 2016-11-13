package uk.ac.ucl.cs.mr.statnlpbook.assignment2

import scala.util.Random

/**
 * Created by Georgios on 16/11/2015.
 */
case class ErrorAnalysis(candidates:Seq[Candidate], golds:Seq[Label], preds:Seq[Label]) {
  val rng = new Random(101)
  val wrong = candidates.zip(golds.zip(preds)).filter(v=>v._2._1!=v._2._2 && v._2._1 =="Positive_regulation").toIndexedSeq
//  val wrong = candidates.zip(golds.zip(preds)).filter(v=>v._2._1!=v._2._2).toIndexedSeq

  val labels = golds.distinct
  println("Confusion Matrix: row (true label); column (predicted label)")
  print("\t")
  println(labels.map(_.slice(0,3)).mkString(" \t"))
  for(corr <- labels){
    val row = collection.mutable.ListBuffer.empty[Int]
    for(mypred <- labels){
      row += candidates.zip(golds.zip(preds)).count(v=>v._2._1==corr && v._2._2 ==mypred)
    }
    print(corr.slice(0,3))
    print("\t")
    if (corr == "None")
      println(row.mkString("   \t"))
    else
      println(row.mkString("\t\t"))
  }
  /**
   * Gives a brief report of mislabelling errors
   * @param num number of mislabelled examples to show
   */
  def showErrors(num:Int=1): Unit = for (_ <- 0 until num) {
    def prettyText(sentence: Sentence, begin:Int, end:Int, lineLength:Int = 1000)={
      val sb = new StringBuilder()
      for (i <- begin until end) {
        val word = sentence.tokens(i).word
        sb.append(word)
        if (i!=end-1) sb.append(" ")
        if ((i-begin) % lineLength == lineLength-1) sb.append("\n")
      }
      sb.toString
    }

    val idx = rng.nextInt(wrong.length)
    val candidate = wrong(idx)._1
    val gold = wrong(idx)._2._1
    val pred = wrong(idx)._2._2
    val isArgument = (candidate.parentIndex > -1)
    val candidateType = if (isArgument) "Argument" else "Trigger"
    val thisSentence = candidate.doc.sentences(candidate.sentenceIndex)
    val candidateText = prettyText(thisSentence,candidate.begin,candidate.end)
    val sentenceText = prettyText(thisSentence,0,thisSentence.tokens.length,20)
    println(s"${candidateType} candidate '${candidateText}' with true label '${gold}' mislabelled as '${pred}'")
    if (isArgument){
      val trigger = thisSentence.events(candidate.parentIndex)
      val triggerText =  prettyText(thisSentence,trigger.begin, trigger.end)
      println(s"Argument belongs to event with trigger '${triggerText}'")
    }
    println(s"Found in file '${candidate.doc.fileName}', in sentence:")
    println(s"${sentenceText}")
    println()
  }



}
