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
    public SingleNeuron(){
        this.numinstances = 3;
        this.numinputs = 4;//termasuk bias
        
        inputs.addElement(1.0);//bias
        inputs.addElement(1.0);
        inputs.addElement(0.0);
        inputs.addElement(1.0);
        data.addElement(inputs);
        
        inputs.addElement(1.0);//bias
        inputs.addElement(0.0);
        inputs.addElement(-1.0);
        inputs.addElement(-1.0);
        data.addElement(inputs);
        
        inputs.addElement(1.0);//bias
        inputs.addElement(-1.0);
        inputs.addElement(-0.5);
        inputs.addElement(-1.0);
        data.addElement(inputs);
        
        weights.addElement(0.0);
        weights.addElement(0.0);
        weights.addElement(0.0);
        weights.addElement(0.0);   
        
        targets.addElement(-1.0);
        targets.addElement(1.0);
        targets.addElement(1.0);
    }
}
