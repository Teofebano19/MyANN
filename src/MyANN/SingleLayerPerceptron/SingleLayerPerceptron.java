package MyANN.SingleLayerPerceptron;



import java.util.Vector;
import weka.classifiers.Classifier;
import weka.core.Instances;

/**
 *
 * @author Andrey
 */
public abstract class SingleLayerPerceptron extends Classifier{
    public int numinstances;
    public int numinputs;
    public Vector<Double> weights;
    public Vector<Double> inputs;
    public Vector<Double> targets;
    public Vector<Vector<Double>> data;
    
    // CTOR
    public SingleLayerPerceptron(){
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
        
        weights.addElement(0.0);
        weights.addElement(0.0);
        weights.addElement(0.0);
        weights.addElement(0.0);   
        
        targets.addElement(-1.0);
        targets.addElement(1.0);
        targets.addElement(1.0);
    }
}
