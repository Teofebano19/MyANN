package SinglePerceptron;

import java.util.Vector;

/**
 *
 * @author Andrey
 */
public class SingleNeuron {
    public int numinstances;
    public int numinputs;
    public Vector<Double> weights;
    public Vector<Double> inputs;
    public Vector<Double> targets;
    public Vector<Vector<Double>> data;
    
    // CTOR
    SingleNeuron(){
        this.numinstances = 3;
        this.numinputs = 4;//termasuk bias
        
        inputs.add(1.0);//bias
        inputs.add(1.0);
        inputs.add(0.0);
        inputs.add(1.0);
        data.add(inputs);
        
        inputs.add(1.0);//bias
        inputs.add(0.0);
        inputs.add(-1.0);
        inputs.add(-1.0);
        data.add(inputs);
        
        inputs.add(1.0);//bias
        inputs.add(-1.0);
        inputs.add(-0.5);
        inputs.add(-1.0);
        data.add(inputs);
        
        weights.add(0.0);
        weights.add(0.0);
        weights.add(0.0);
        weights.add(0.0);   
        
        targets.add(-1.0);
        targets.add(1.0);
        targets.add(1.0);
    }
}
