# NULLQUAKE™® MODEL™®

## FEATURE VECTOR
We start by parsing the qasm file into a feature vector. This contains 15 features:
$$
\begin{bmatrix}
\text{Depth} \\
\text{X Gates} \\
\text{Qubits that have X Gates} \\
\text{Z Gates} \\
\text{Qubits that have Z Gates} \\
\text{Cx Gates} \\
\text{Qubits that have Cx Gates} \\
\text{H Gates} \\
\text{Qubits that have H Gates} \\
\text{T Gates} \\
\text{Total Number of Qbits} \\
\text{Topology} \\
\text{Qubit nodes} \\
\text{number of different gates} \\
\end{bmatrix}
$$

During this process we also take into account combinations that the compiler would have already optimized out, such as gates canceling.

# Models:

Base model: 8.9%

GPT: 79.97%

GPT2: 82.5%
