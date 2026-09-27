NULLQUAKE

This is our submission for the SCQuantathonV3 Quantum Rings challange.

Problem Statement:

Build a model to predict the runtime of a quantum circuit in the QuantumRings simulator, and do it all without spending more than 15 seconds on an output. We were given 535 circuits, yielding 1596 different data points for training. Each datapoint had a qasm file, a threshold, and the total runtime.

Approach:

We all had no prior experience with Machine Learning, so our first objective was to research and learn how to make a model.

We decided we had two main tasks: design a feature space, and then choose a good model to train.

We decided to go for a maximal feature space because there were so many files so we had no idea which features were most important. Also we wanted to have a very generalized model. So our original feature space had 94 features to cover as many bases as possible. These features included things like the number of each possible gate, the number of qubits, the depth of the circuit, the Paralelism, the graph density, and many more.
As for deciding on the model, once again we decided to throw a braod net and tested as many different types of models as possible, and compared the outputs.
The models we tried were Linear regression, KNN, Gradient Boosted Trees, Random Forests, Support Vector Machines, XG Boost, and combined ensemble methods. The combined ensemble won.

