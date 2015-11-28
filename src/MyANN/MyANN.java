package MyANN;

import MyANN.neurons.Neuron;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;

public class MyANN extends Classifier {
    
    private List<Layer> layers;
    private final List<Integer> layersSize;
    private boolean isUpdatingPerEpoch;
    private int nEpoch = 100;
    private int nLayers;
    private double learningRate = 0.1;
    private double momentum = 0.2;
    private double errorThreshold = 0.01;
    private Neuron neuron;

    public MyANN(List<Integer> layersSize, Neuron neuron) {
        this.layersSize = layersSize;
        this.nLayers = layersSize.size() + 1;
        this.neuron = neuron;
    }
    
    @Override
    public void buildClassifier(Instances instances) throws Exception {
        instances.deleteWithMissingClass();
        buildNetwork(instances);
        System.out.println("Finished building network");
        train(instances);
        System.out.println("Finished training");
    }
    
    @Override
    public double classifyInstance(Instance instance) {
        if (nLayers == 0) {
            throw new IllegalStateException("Classifier must be built first to classify instance");
        }
        List<Double> input = getInput(instance);
        List<Double> lastOutput = input;
        for (Layer layer : layers) {
            lastOutput = layer.processInput(lastOutput);
            System.out.println("lastOutput: " + lastOutput);
        }
        int argmax = 0;
        for (int i = 1; i < lastOutput.size(); ++i) {
            if (lastOutput.get(i) > lastOutput.get(argmax)) {
                argmax = i;
            }
        }
        return (double)argmax;
   }

    private void buildNetwork(Instances instances) {
        // TODO: handle nominal attributes
        int numInput = 0;
        for (int i = 0; i < instances.numAttributes(); ++i) {
            if (i == instances.classIndex()) {
                continue;
            }
            numInput++;
        }
        layers = new ArrayList<>();
        int numLastLayer = numInput;
        // TODO: custom weight initialization
        Random random = new Random(0);
        for (Integer layerSize : layersSize) {
            Layer layer = new Layer(numLastLayer, layerSize, neuron);
//            layer.setRandomWeight(random);
            layers.add(layer);
            numLastLayer = layerSize;
        }
        // final layer
        Layer lastLayer = new Layer(numLastLayer, instances.classAttribute().numValues(), neuron);
//        lastLayer.setRandomWeight(random);
        layers.add(lastLayer);
    }

    private void train(Instances instances) {
        for (int epoch = 0; epoch < nEpoch; ++epoch) {
            double totalError = 0;
            for (int iInstance = 0; iInstance < instances.numInstances(); ++iInstance) {
                Instance instance = instances.instance(iInstance);
                List<Double> input = getInput(instance);
                List<Double> expected = getExpected(instance);
                List<List<Double>> outputs = new ArrayList<>();
                List<Double> lastOutput = input;
                outputs.add(lastOutput);
                for (Layer layer : layers) {
                    List<Double> output = layer.processInput(lastOutput);
                    outputs.add(output);
                    lastOutput = output;
                }
                double error = calculateError(expected, lastOutput);
                totalError += error;
                System.out.println("\t" + iInstance + ": error = " + error);
                // back propagation starts here
                List<List<Double>> deltas = new ArrayList<>();
                for (int i = 0; i < nLayers; ++i) {
                    deltas.add(null);
                }
                deltas.set(nLayers - 1, layers.get(nLayers - 1).calculateDelta(expected, lastOutput));
                for (int i = nLayers - 2; i >= 0; --i) {
                    deltas.set(i, layers.get(i).calculateDeltaWithBackpropagation(outputs.get(i + 1), layers.get(i), deltas.get(i + 1)));
                }
                for (int i = nLayers - 1; i >= 0; --i) {
                    layers.get(i).updateWeights(learningRate, momentum, outputs.get(i), deltas.get(i));
                }
            }
            System.out.println("Epoch " + epoch + ": error = " + totalError);
            System.out.println("===========================================");
            if (totalError < errorThreshold) {
                break;
            }
        }
    }

    private List<Double> getInput(Instance instance) {
        // TODO: handle nominal attributes
        List<Double> input = new ArrayList<>();
        for (int i = 0; i < instance.numAttributes(); ++i) {
            if (i == instance.classIndex()) {
                continue;
            }
            input.add(instance.value(i));
        }
        return input;
    }

    private List<Double> getExpected(Instance instance) {
        List<Double> input = new ArrayList<>();
        for (int i = 0; i < instance.numClasses(); ++i) {
            if (i == (int)instance.classValue()) {
                input.add(1.0);
            } else {
                input.add(0.0);
            }
        }
        return input;
    }

    private double calculateError(List<Double> expected, List<Double> outputs) {
        double error = 0;
        for (int i = 0; i < expected.size(); ++i) {
            double target = expected.get(i);
            double output = outputs.get(i);
            error += 0.5 * (target - output) * (target - output);
        }
        return error;
    }
    
}
