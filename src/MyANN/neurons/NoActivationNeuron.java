package MyANN.neurons;

public class NoActivationNeuron extends Neuron{

    @Override
    public double calculateActivation(double input) {
        return input;
    }

    @Override
    public double calculateDelta(double output) {
        return 1;
    }
    
}
