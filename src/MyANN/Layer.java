package MyANN;

import MyANN.neurons.Neuron;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class Layer implements Serializable {
    
    private static final long serialVersionUID = 1L;
    
    private final int inputSize;
    private final int nNeurons;
    private final List<Neuron> neurons = new ArrayList<>();
    private final List<List<Double>> weights = new ArrayList<>();
    private final List<List<Double>> lastDeltaWeights = new ArrayList<>();
    private static final Double BIAS_VALUE = 1.0;
    
    public Layer(int inputSize, int nNeurons, Neuron neuron) {
        this.inputSize = inputSize;
        this.nNeurons = nNeurons;
        // TODO: customizable/random initial weights
        for (int i = 0; i < nNeurons; ++i) {
            List<Double> currentWeights = new ArrayList<>();
            List<Double> currentLastDeltaWeights = new ArrayList<>();
            for (int j = 0; j <= inputSize; ++j) {
                currentWeights.add(0.);
                currentLastDeltaWeights.add(0.);
            }
            weights.add(currentWeights);
            lastDeltaWeights.add(currentLastDeltaWeights);
            neurons.add(neuron);
        }
    }
    
    public void setRandomWeight(Random random) {
        for (int i = 0; i < nNeurons; ++i) {
            for (int j = 0; j <= inputSize; ++j) {
                weights.get(i).set(j, random.nextDouble());
            }
        }
    }
    
    public void setWeight(int from, int to, double value) {
        weights.get(to).set(from, value);
    }
    
    public void setNeuron(int index, Neuron neuron) {
        neurons.set(index, neuron);
    }
    
    public int size() {
        return nNeurons;
    }
    
    public int inputSize() {
        return inputSize;
    }
    
    /**
     * @param input the input without bias. Bias is always set to 1.
     */
    public List<Double> processInput(List<Double> input) {
        List<Double> inputWithBias = getInputWithBias(input);
        List<Double> output = new ArrayList<>();
        for (int i = 0; i < neurons.size(); ++i) {
            output.add(neurons.get(i).processInput(weights.get(i), inputWithBias));
        }
//        System.out.println("input: " + input + "\noutput: " + output);
        return output;
    }
    
    public List<Double> calculateDelta(List<Double> targets, List<Double> outputs) {
        if (targets.size() != neurons.size() || outputs.size() != neurons.size()) {
            throw new IllegalArgumentException(
                    "Target or output size mismatch, expecting " + 
                            neurons.size() + ", target was " + targets.size() + 
                            " and output was " + outputs.size());
        }
        List<Double> deltas = new ArrayList<>();
        for (int i = 0; i < targets.size(); ++i) {
            double target = targets.get(i);
            double output = outputs.get(i);
            deltas.add((target - output) * neurons.get(i).calculateDelta(output));
        }
//        System.out.println("target: " + targets + "\noutput: " + outputs + "\ndelta: " + deltas);
        return deltas;
    }
    
    public List<Double> calculateDeltaWithBackpropagation(List<Double> outputs, Layer frontLayer, List<Double> frontDeltas) {
        if (outputs.size() != neurons.size()) {
            throw new IllegalArgumentException(
                    "Output size mismatch, expecting " + neurons.size() + ", was " + outputs.size());
        }
        List<Double> deltas = new ArrayList<>();
        for (int i = 0; i < outputs.size(); ++i) {
            double propagatedDelta = 0;
            double output = outputs.get(i);
            for (int j = 0; j < frontLayer.nNeurons; ++j) {
//                System.out.printf("i = %d, j = %d, nNeuron = %d, frontDeltas = %d\n", i, j, frontLayer.nNeurons, frontDeltas.size());
                propagatedDelta += frontLayer.weights.get(j).get(i) * frontDeltas.get(j);
            }
            deltas.add(propagatedDelta * neurons.get(i).calculateDelta(output));
        }
//        System.out.println("delta: " + deltas + "\n");
        return deltas;
    }
    
    public void updateWeights(double learningRate, double momentum, List<Double> inputs, List<Double> deltas) {
        List<Double> inputWithBias = getInputWithBias(inputs);
        for (int from = 0; from <= inputSize; ++from) {
            for (int to = 0; to < nNeurons; ++to) {
                lastDeltaWeights.get(to).set(from, learningRate * deltas.get(to) * inputWithBias.get(from) + lastDeltaWeights.get(to).get(from) * momentum);
                weights.get(to).set(from, weights.get(to).get(from) + lastDeltaWeights.get(to).get(from));
            }
        }
//        System.out.println("Weights updated!\n" + weights);
    }

    private List<Double> getInputWithBias(List<Double> input) {
        if (inputSize != input.size()) {
            throw new IllegalArgumentException(
                    "Input size mismatch, expecting " + inputSize + ", was " + input.size());
        }
        List<Double> inputWithBias = new ArrayList<>();
        inputWithBias.addAll(input);
        inputWithBias.add(BIAS_VALUE);
        return inputWithBias;
    }
}
