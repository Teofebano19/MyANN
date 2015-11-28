package MyANN.neurons;

import java.io.Serializable;
import java.util.List;
import java.util.function.Function;
import javafx.util.Pair;

public abstract class Neuron implements Serializable {
    
    private static final long serialVersionUID = 1L;
    
    public double processInput(List<Double> weight, List<Double> input) {
        if (weight.size() != input.size()) {
            throw new IllegalArgumentException(
                    "weight and input must contain the same number of elements");
        }
        double realInput = 0;
        for (int i = 0; i < weight.size(); ++i) {
            realInput += weight.get(i) * input.get(i);
        }
        return calculateActivation(realInput);
    }
    
    public abstract double calculateActivation(double input);
    
    public abstract double calculateDelta(double output);
    
}
