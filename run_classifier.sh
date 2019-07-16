#!/bin/bash

mkdir -p "./classifier/output"

LOG=classifier/output/run.log
ERROR=classifier/output/error.log

preprocess_and_learn() {

  echo Preprocessing $1 $2 >> $LOG
  python3 classifier/preprocess/preprocess.py -s $1 -g $2 2>>$ERROR 1>>$LOG
  echo $'\n' >> $LOG

  echo NeuralNetwork $1 $2 \(Normal\) >> $LOG
  python3 classifier/learn/learner.py -a NeuralNetworkAlgorithm 2>>$ERROR 1>>$LOG
  echo $'\n' >> $LOG

  echo NeuralNetwork $1 $2 \(Undersampled\) >> $LOG
  python3 classifier/learn/learner.py -a NeuralNetworkAlgorithm -u Undersampling 2>>$ERROR 1>>$LOG
  echo $'\n' >> $LOG

  echo NeuralNetwork $1 $2 \(Oversampled\) >> $LOG
  python3 classifier/learn/learner.py -a NeuralNetworkAlgorithm -u Oversampling 2>>$ERROR 1>>$LOG
  echo $'\n' >> $LOG

  echo RandomForest $1 $2 \(Normal\) >> $LOG
  python3 classifier/learn/learner.py -a RandomForestClassifierAlgorithm 2>>$ERROR 1>>$LOG
  echo $'\n' >> $LOG

  echo RandomForest $1 $2 \(Undersampled\) >> $LOG
  python3 classifier/learn/learner.py -a RandomForestClassifierAlgorithm -u Undersampling 2>>$ERROR 1>>$LOG
  echo $'\n' >> $LOG

  echo RandomForest $1 $2 \(Oversampled\) >> $LOG
  python3 classifier/learn/learner.py -a RandomForestClassifierAlgorithm -u Oversampling 2>>$ERROR 1>>$LOG
  echo $'\n' >> $LOG

}

preprocess_and_learn_all_groupers() {

  echo Learning on APTGrouper >> $LOG
  preprocess_and_learn $1 APTGrouper

  echo Learning on CountryGrouper >> $LOG
  preprocess_and_learn $1 CountryGrouper

  echo Learning on CountrySeparatedGroupAndFamiliesGrouper >> $LOG
  preprocess_and_learn $1 CountrySeparatedGroupAndFamiliesGrouper

}

preprocess_and_learn_all_groupers_and_selectors() {

  echo Learning on CuckooExtractedSelector >> $LOG
  preprocess_and_learn_all_groupers CuckooExtractedSelector
  
  echo Learning on CuckooFilteredSelector >> $LOG
  preprocess_and_learn_all_groupers CuckooFilteredSelector
  
}

preprocess_and_learn_all_groupers_and_selectors

echo Calculating all results
python3 classifier/results/calculate_results.py > classifier/results/tables.tex

