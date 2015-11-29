package MyANN.neurons;

public class SignNeuron extends Neuron{

    @Override
    public double calculateActivation(double input) {
        if(input < 0)
            return -1;
        else
            return 1;
    }

    @Override
    public double calculateDelta(double output) {
        return 1;
    }
    
}
