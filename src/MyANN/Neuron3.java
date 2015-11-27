package MyANN;

import java.util.List;
import java.util.function.Function;
import javafx.util.Pair;

public class Neuron3 {
    
    private final Function<Double, Double> activation;
    private final Function<Pair<Double, Double>, Double> error;
    
    public Neuron3(Function<Double, Double> activation, Function<Pair<Double, Double>, Double> error) {
        this.activation = activation;
        this.error = error;
    }
    
    public double processInput(List<Double> weight, List<Double> input) {
        if (weight.size() != input.size()) {
            throw new IllegalArgumentException(
                    "weight and input must contain the same number of elements");
        }
        double realInput = 0;
        for (int i = 0; i < weight.size(); ++i) {
            realInput += weight.get(i) * input.get(i);
        }
        return activation.apply(realInput);
    }
    
    public double calculateError(double output, double target) {
        return error.apply(new Pair<Double, Double>(output, target));
    }
    
}
