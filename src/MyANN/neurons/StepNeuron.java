package MyANN.neurons;

public class StepNeuron extends Neuron{

    @Override
    public double calculateActivation(double input) {
        if(input < 0)
            return 0;
        else
            return 1;
    }

    @Override
    public double calculateDelta(double output) {
        return output * (1 - output);
    }
}
