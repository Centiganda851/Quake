# NULLQUAKE™® MODEL™®

## FEATURE VECTOR
We start by parsing the qasm file into a feature vector. This contains 15 features: 
Depth,
X Gates,
Qubits that have X Gates,
Z Gates,
Qubits that have Z Gates,
Cx Gates,
Qubits that have Cx Gates,
H Gates,
Qubits that have H Gates,
T Gates,
Total Number of Qbits,
Topology,
Qubit nodes,
number of different gates

During this process we also take into account combinations that the compiler would have already optimized out, such as gates canceling.

# Models:

Base model: 8.9%

GPT: 79.97%

GPT2: 82.5%

NeuralNet1: 77.27%

KNN: 93.74%

Extra trees\* 89.2%

SVM: 78.41%

Random Forest: 88.49%


\* trained on the full data set
