find "/home/pasricha/jaan/lib/python3.7/site-packages/tensorflow_core"  -type f -name "*.py" -mtime -15 -exec ls -l {} \; >> Last15DayCahnge.txt
find "/home/pasricha/jaan/keras_log-weights/"  -type f -name "*.py" -mtime -15 -exec ls -l {} \; >> Last15DayCahnge.txt
find "/home/pasricha/WeightPredictions/"  -type f -name "*.py" -mtime -15 -exec ls -l {} \; >> Last15DayCahnge.txt
