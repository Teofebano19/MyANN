package MyANN.neurons;

public class SigmoidNeuron extends Neuron {

    @Override
    public double calculateActivation(double input) {
        return (1 / (1 + Math.exp(-input)));
    }

    @Override
    public double calculateDelta(double output) {
        return output * (1 - output);
    }
    
}
