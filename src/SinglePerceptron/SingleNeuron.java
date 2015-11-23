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
        weights = new Vector<>();
        inputs = new Vector<>();
        targets = new Vector<>();
        data = new Vector<>();
        
        this.numinstances = 3;
        this.numinputs = 4;//termasuk bias
        
        inputs.addElement(1.0);//bias
        inputs.addElement(1.0);
        inputs.addElement(0.0);
        inputs.addElement(1.0);
        data.addElement(inputs);
        
        inputs = new Vector<>();
        inputs.addElement(1.0);//bias
        inputs.addElement(0.0);
        inputs.addElement(-1.0);
        inputs.addElement(-1.0);
        data.addElement(inputs);
        
        inputs = new Vector<>();
        inputs.addElement(1.0);//bias
        inputs.addElement(-1.0);
        inputs.addElement(-0.5);
        inputs.addElement(-1.0);
        data.addElement(inputs);
        
        System.out.println(data.elementAt(0).elementAt(0)+" , "+data.elementAt(0).elementAt(1)+" , "+data.elementAt(0).elementAt(2)+" , "+data.elementAt(0).elementAt(3));
        System.out.println(data.elementAt(1).elementAt(0)+" , "+data.elementAt(1).elementAt(1)+" , "+data.elementAt(1).elementAt(2)+" , "+data.elementAt(1).elementAt(3));
        System.out.println(data.elementAt(2).elementAt(0)+" , "+data.elementAt(2).elementAt(1)+" , "+data.elementAt(2).elementAt(2)+" , "+data.elementAt(2).elementAt(3));
                
        weights.addElement(0.0);
        weights.addElement(0.0);
        weights.addElement(0.0);
        weights.addElement(0.0);   
        
        targets.addElement(-1.0);
        targets.addElement(1.0);
        targets.addElement(1.0);
    }
}
