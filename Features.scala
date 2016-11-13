package uk.ac.ucl.cs.mr.statnlpbook.assignment2

import uk.ac.ucl.cs.mr.statnlpbook.assignment2._

import scala.collection.mutable

/**
 * Created by Georgios on 05/11/2015.
 */

object Features {

  /**
   * a feature function with two templates w:word,label and l:label.
   * Example for Trigger Exraction
   * @param x
   * @param y
   * @return
   */
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
  /**
   * a feature function with two templates w:word,label and l:label.
   * Example for Argument Exraction
   * @param x
   * @param y
   * @return
   */
  def defaultArgumentFeatures(x: Candidate, y: Label): FeatureVector = {
    val doc = x.doc
    val begin = x.begin
    val end = x.end
    val thisSentence = doc.sentences(x.sentenceIndex)
    val event = thisSentence.events(x.parentIndex) //use this to gain access to the parent event
    val eventHeadToken = thisSentence.tokens(event.begin) //first token of event
    val feats = new mutable.HashMap[FeatureKey,Double]
    feats += FeatureKey("label bias", List(y)) -> 1.0
    val token = thisSentence.tokens(begin) //first word of argument
    feats += FeatureKey("first argument word", List(token.word, y)) -> 1.0
    feats += FeatureKey("is protein_first trigger word", List(x.isProtein.toString,eventHeadToken.word, y)) -> 1.0
    feats.toMap
  }


  //TODO: make your own feature functions
  def myTriggerFeatures(x: Candidate, y: Label): FeatureVector = {
     val doc = x.doc
     val begin = x.begin
     val end = x.end
     val thisSentence = doc.sentences(x.sentenceIndex) //use this to gain access to the parent sentence
     val feats = new mutable.HashMap[FeatureKey,Double]
     feats += FeatureKey("label bias", List(y)) -> 1.0 //bias feature
     val token = thisSentence.tokens(begin) //first token of Trigger
     feats += FeatureKey("first trigger word", List(token.word, y)) -> 1.0 //word feature
     // get token features

     feats += FeatureKey("token POS", List(token.pos, y)) -> 1.0 // the POS of token
     if (token.word.capitalize == token.word)
        feats += FeatureKey("Capitialization", List(token.word, y)) -> 1.0 // Capitialization


     // get mention features
     val mention_set = collection.mutable.Set[Int]()
     thisSentence.mentions.foreach(x => for (i <- x.begin to x.end) mention_set += i)

     val protein_mention = thisSentence.mentions.filter(_.label == "Protein")
     feats += FeatureKey("number of mentioned protein", List(y)) -> 1.0 * protein_mention.length // number of mentioned protein in the sentence

     val mention_variety_lst = thisSentence.mentions.map(_.label)
     val mention_variety = mention_variety_lst.distinct.length
     feats += FeatureKey("number of different types of mention", List(y)) -> 1.0 * mention_variety

     if (protein_mention.nonEmpty){
        // cloest The mentioned protein name in the front
        val unfound_ahead = Mention("unfound", -1, -1)
        val close_ahead_mention = protein_mention.foldLeft(unfound_ahead)( (nearest:Mention, next:Mention) => {
           val current_best = nearest.begin - begin
           val next_dist = next.begin - begin
           if (next_dist < 0 && next_dist > current_best) next else nearest
        })
        if (close_ahead_mention != unfound_ahead){
           val ind = close_ahead_mention.begin
           feats += FeatureKey("nearest ahead protein name", List(thisSentence.tokens(ind).word, y)) -> 1.0
        }
        // cloest The mentioned protein name behind
        val unfound_behind = Mention("unfound", 1000000, 1000000)
        val close_behind_mention = protein_mention.foldRight(unfound_behind)( (nearest:Mention, next:Mention) => {
           val current_best = nearest.begin - begin
           val next_dist = next.begin - begin
           if (next_dist > 0 && next_dist < current_best) next else nearest
        })
        if (close_behind_mention != unfound_behind){
           val ind = close_behind_mention.begin
           feats += FeatureKey("nearest behind protein name", List(thisSentence.tokens(ind).word, y)) -> 1.0
        }
     }


     // dependency features
     // extract all dependency structures from candidate
     val candidate_bgn_dep = thisSentence.deps.filter(x => x.mod == begin)
     candidate_bgn_dep.foreach(x => {
        val headwd = thisSentence.tokens(x.head).word
        feats += FeatureKey("candidate bgn dep", List(x.label, headwd, y)) -> 1.0
        if (headwd == headwd.capitalize)
           feats += FeatureKey("candidate bgn dep cap", List(x.label, y)) -> 1.0
     })

     val candidate_head_dep = thisSentence.deps.filter(x => x.head == begin)
     candidate_head_dep.foreach(x => {
        val mod = thisSentence.tokens(x.mod).word
        feats += FeatureKey("candidate head dep", List(x.label, mod, y)) -> 1.0
        if (mod == mod.capitalize)
           feats += FeatureKey("candidate head dep cap", List(x.label, y)) -> 1.0
     })
     // subject phrase if candidate is object
     if(candidate_bgn_dep.exists(_.label == "dobj")){
        val subject = thisSentence.deps.filter(_.label == "nsubj")
        if (subject.nonEmpty){
           val subj_ind = subject.map(_.mod).toSet
           val subj_nn = thisSentence.deps.filter(x => subj_ind.contains(x.head) && x.label == "nn")
           val subj_whole = subject ::: subj_nn
           subj_whole.foreach(x => feats += FeatureKey("subj_nn", List(thisSentence.tokens(x.mod).word, y)) -> 1.0)
        }
     }

     // prepof + noun phrase (hoc 2)
     // PMID-8077662.json The involvement
     val prepof = candidate_head_dep.filter(_.label=="prep_of")
     if (prepof.nonEmpty){
        val tok_ind_prepof = prepof.map(_.mod).toSet
        val nn_hoc2 = thisSentence.deps.filter(x => tok_ind_prepof.contains(x.head) && x.label == "nn")
        nn_hoc2.foreach(x => feats += FeatureKey("prepof_nn", List(thisSentence.tokens(x.mod).word, y)) -> 1.0)
     }



     feats.toMap
  }


  def myArgumentFeatures(x: Candidate, y: Label): FeatureVector = {
     val doc = x.doc
     val begin = x.begin
     val end = x.end
     val thisSentence = doc.sentences(x.sentenceIndex)
     val event = thisSentence.events(x.parentIndex) //use this to gain access to the parent event
     val eventHeadToken = thisSentence.tokens(event.begin) //first token of event
     val feats = new mutable.HashMap[FeatureKey,Double]
     feats += FeatureKey("label bias", List(y)) -> 1.0
     val token = thisSentence.tokens(begin) //first word of argument
     feats += FeatureKey("first argument word", List(token.word, y)) -> 1.0
     feats += FeatureKey("is protein_first trigger word", List(x.isProtein.toString,eventHeadToken.word, y)) -> 1.0

     //token feature
     feats += FeatureKey("is protein_first argument word", List(x.isProtein.toString,token.word, y)) -> 1.0
     feats += FeatureKey("first argument word capital", List((token.word == token.word.capitalize).toString, token.word, y)) -> 1.0
     feats += FeatureKey("first event word capital", List((eventHeadToken.word == eventHeadToken.word.capitalize).toString, eventHeadToken.word, y)) -> 1.0
     feats += FeatureKey("first argument word pos", List(token.pos, y)) -> 1.0
     feats += FeatureKey("first event word pos", List(eventHeadToken.pos, y)) -> 1.0
     //extract the pos pair feature of the argument and the parent event of this argument
     feats += FeatureKey("first argument and event word pos pair", List(token.pos, eventHeadToken.pos, y)) -> 1.0


     //mention feature
     val protein_mention = thisSentence.mentions.filter(_.label == "Protein")
     feats += FeatureKey("number of mentioned protein", List(y)) -> 1.0 * protein_mention.length // number of mentioned protein in the sentence


     if (protein_mention.nonEmpty) {
        protein_mention.foreach({x =>
           if((x.begin < begin && x.end > begin) || (x.begin < end && x.end > end)) {
              feats += FeatureKey("argument of protein", List(x.label, event.gold, y)) -> 1.0
           }
        })
     }

     if (protein_mention.nonEmpty){
        // cloest The mentioned protein name in the front
        val unfound_ahead = Mention("unfound", -1, -1)
        val close_ahead_mention = protein_mention.foldLeft(unfound_ahead)( (nearest:Mention, next:Mention) => {
           val current_best = nearest.begin - begin
           val next_dist = next.begin - begin
           if (next_dist < 0 && next_dist > current_best) next else nearest
        })
        if (close_ahead_mention != unfound_ahead){
           val ind = close_ahead_mention.begin
           feats += FeatureKey("nearest ahead protein name", List(thisSentence.tokens(ind).word, y)) -> 1.0
        }
        // cloest The mentioned protein name in the front in the parent event
        val close_ahead_mention_PEvent = protein_mention.foldLeft(unfound_ahead)( (nearest:Mention, next:Mention) => {
           val current_best = nearest.begin - event.begin
           val next_dist = next.begin - event.begin
           if (next_dist < 0 && next_dist > current_best) next else nearest
        })

        if (close_ahead_mention_PEvent != unfound_ahead){
           feats += FeatureKey("nearest ahead protein name of the parent event", List(thisSentence.tokens(close_ahead_mention_PEvent.begin).word, y)) -> 1.0
        }

        // cloest The mentioned protein name behind
        val unfound_behind = Mention("unfound", 1000000, 1000000)
        val close_behind_mention = protein_mention.foldRight(unfound_behind)( (nearest:Mention, next:Mention) => {
           val current_best = nearest.begin - begin
           val next_dist = next.begin - begin
           if (next_dist > 0 && next_dist < current_best) next else nearest
        })
        if (close_behind_mention != unfound_behind){
           val ind = close_behind_mention.begin
           feats += FeatureKey("nearest behind protein name", List(thisSentence.tokens(ind).word, y)) -> 1.0
        }
        // cloest The mentioned protein name behind in parent event
        val close_behind_mention_PEvent = protein_mention.foldRight(unfound_behind)( (nearest:Mention, next:Mention) => {
           val current_best = nearest.begin - event.begin
           val next_dist = next.begin - event.begin
           if (next_dist > 0 && next_dist < current_best) next else nearest
        })

        if (close_behind_mention_PEvent != unfound_behind) {
           feats += FeatureKey("nearest behind protein name of the parent event", List(thisSentence.tokens(close_behind_mention_PEvent.begin).word, y)) -> 1.0
        }
     }

     //dependency

     val candidate_bgn_dep = thisSentence.deps.filter(x => x.mod == begin)

      // Dependency and parent event feature
      candidate_bgn_dep.foreach({x =>
         if((event.begin <= x.head) && (x.head <= event.end)) {
            feats += FeatureKey("dep and parent event", List(x.label, event.gold, y)) -> 1.0
         }
      })
      //extract head dependencies that have the modifier as the candidate
     candidate_bgn_dep.foreach(x => {
        val headwd = thisSentence.tokens(x.head).word
        feats += FeatureKey("candidate bgn dep", List(x.label, headwd, y)) -> 1.0
        if (headwd == headwd.capitalize)
           feats += FeatureKey("candidate bgn dep cap", List(x.label, y)) -> 1.0
     })
     //extract head dependencies that have the modifier as the parent event of the candidate
     val candidate_bgn_dep_PEvent = thisSentence.deps.filter(x => x.mod == event.begin)
     candidate_bgn_dep_PEvent.foreach(x => {
        val headwd = thisSentence.tokens(x.head).word
        feats += FeatureKey("candidate bgn dep in parent event", List(x.label, headwd, y)) -> 1.0
        if (headwd == headwd.capitalize)
           feats += FeatureKey("candidate bgn dep cap in parent event", List(x.label, y)) -> 1.0
     })
     //extract modifier dependencies that have the modifier as the candidate

     val candidate_head_dep = thisSentence.deps.filter(x => x.head == begin)
     candidate_head_dep.foreach(x => {
        val mod = thisSentence.tokens(x.mod).word
        feats += FeatureKey("candidate head dep", List(x.label, mod, y)) -> 1.0
        if (mod == mod.capitalize)
           feats += FeatureKey("candidate head dep cap", List(x.label, y)) -> 1.0
     })
     //extract modifier dependencies that have the modifier as the parent of the s candidate
     val candidate_head_dep_PEvent = thisSentence.deps.filter(x => x.head == event.begin)
     candidate_head_dep_PEvent.foreach(x => {
        val mod = thisSentence.tokens(x.mod).word
        feats += FeatureKey("candidate head dep in parent event", List(x.label, mod, y)) -> 1.0
        if (mod == mod.capitalize)
           feats += FeatureKey("candidate head dep cap in parent event", List(x.label, y)) -> 1.0
     })

     //subject phrase if candidate is object
     if(candidate_bgn_dep.exists(_.label == "dobj")){
        val subject = thisSentence.deps.filter(_.label == "nsubj")
        if (subject.nonEmpty){
           val subj_ind = subject.map(_.mod).toSet
           val subj_nn = thisSentence.deps.filter(x => subj_ind.contains(x.head) && x.label == "nn")
           val subj_whole = subject ::: subj_nn
           subj_whole.foreach(x => feats += FeatureKey("subj_nn", List(thisSentence.tokens(x.mod).word, y)) -> 1.0)
        }
     }

     //mix of mentions and dependency

     candidate_bgn_dep.foreach({x =>
        val dependencyMention = protein_mention.filter(i => (i.begin <= x.head) && (i.end >= x.head))
        dependencyMention.foreach(j =>
        feats += FeatureKey("dependency and Mention", List(x.label, j.label, y)) -> 1.0
        )
     })


     // prepof + noun phrase (hoc 2)
     // PMID-8077662.json The involvement
//     val prepof = candidate_head_dep.filter(_.label=="prep_of")
//     if (prepof.nonEmpty){
//        val tok_ind_prepof = prepof.map(_.mod).toSet
//        val nn_hoc2 = thisSentence.deps.filter(x => tok_ind_prepof.contains(x.head) && x.label == "nn")
//        nn_hoc2.foreach(x => feats += FeatureKey("prepof_nn", List(thisSentence.tokens(x.mod).word, y)) -> 1.0)
//     }


     feats.toMap
  }


}
