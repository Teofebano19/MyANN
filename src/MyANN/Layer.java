package MyANN;

import java.util.ArrayList;
import java.util.List;

public class Layer {
    
    private final List<Neuron3> neurons = new ArrayList<>();
    private final List<List<Double>> weights = new ArrayList<>();
    private static final Double BIAS_VALUE = 1.0;
    
    public Layer() {
        // initialize weight here
    }
    
    /**
     * @param input the input without bias. Bias is always set to 1.
     */
    public List<Double> processInput(List<Double> input) {
        List<Double> inputWithBias = new ArrayList<>();
        inputWithBias.addAll(input);
        inputWithBias.add(BIAS_VALUE);
        
        List<Double> output = new ArrayList<>();
        for (int i = 0; i < neurons.size(); ++i) {
            output.add(neurons.get(i).processInput(weights.get(i), inputWithBias));
        }
        return output;
    }
    
}
