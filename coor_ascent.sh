# /data/private/yushi/jdk8u272-b10/bin/java -cp LeToR/RankLib-2.1-patched.jar ciir.umass.edu.features.FeatureManager -input results/cdr_cv_8e.autobert.full_features -output features/ -k 5
/data/private/yushi/jdk8u272-b10/bin/java -jar LeToR/RankLib-2.1-patched.jar -train features/cdr_cv_8e.autobert.full_features -ranker 4 -kcv 5 -kcvmd checkpoints/ -kcvmn ca -metric2t NDCG@3 -metric2T NDCG@3
/data/private/yushi/jdk8u272-b10/bin/java -jar LeToR/RankLib-2.1-patched.jar -load checkpoints/f1.ca -rank features/f1.test.cdr_cv_8e.autobert.full_features -score f1.score
/data/private/yushi/jdk8u272-b10/bin/java -jar LeToR/RankLib-2.1-patched.jar -load checkpoints/f2.ca -rank features/f2.test.cdr_cv_8e.autobert.full_features -score f2.score
/data/private/yushi/jdk8u272-b10/bin/java -jar LeToR/RankLib-2.1-patched.jar -load checkpoints/f3.ca -rank features/f3.test.cdr_cv_8e.autobert.full_features -score f3.score
/data/private/yushi/jdk8u272-b10/bin/java -jar LeToR/RankLib-2.1-patched.jar -load checkpoints/f4.ca -rank features/f4.test.cdr_cv_8e.autobert.full_features -score f4.score
/data/private/yushi/jdk8u272-b10/bin/java -jar LeToR/RankLib-2.1-patched.jar -load checkpoints/f5.ca -rank features/f5.test.cdr_cv_8e.autobert.full_features -score f5.score
python LeToR/gen_trec.py -dev ../ance/result/cdr_cv_8e.auto.full.jsonl -res results/bert_ca.trec -k 5
# rm f1.score
# rm f2.score
