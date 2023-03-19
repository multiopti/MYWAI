using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
//using UnscentedKalmanFilter;
using ZedGraph;

public class ARMA
{
    private int _p;
    private int _q;
    public Vector<double> _phi;
    public Vector<double> _theta;

    public ARMA(int p, int q)
    {
        _p = p;
        _q = q;
        _phi = Vector<double>.Build.Dense(p);
        _theta = Vector<double>.Build.Dense(q);
    }

    public void Fit(Vector<double> data, double lambda = 0.01)
    {
        int n = data.Count;
        Matrix<double> X = Matrix<double>.Build.Dense(n - _p, _p + _q);
        Vector<double> y = Vector<double>.Build.Dense(n - _p);

        for (int i = 0; i < n - _p; i++)
        {
            for (int j = 0; j < _p; j++)
            {
                X[i, j] = data[i + _p - j - 1];
            }
            for (int j = 0; j < _q; j++)
            {
                X[i, j + _p] = -data[i + _p - j];
            }
            y[i] = data[i + _p];
        }

        var xt = X.Transpose();
        var xtx = xt * X;
        var xtxr = xtx + lambda * Matrix<double>.Build.DenseIdentity(_p + _q);
        var xtxi = xtxr.Inverse();
        var xty = xt * y;
        var b = xtxi * xty;

        _phi = b.SubVector(0, _p);
        _theta = b.SubVector(_p, _q);
    }

    public double PredictNext(Vector<double> data)
    {
        int n = data.Count;
        double prediction = 0.0;

        for (int i = 1; i <= _p; i++)
        {
            prediction += _phi[i - 1] * data[n - i];
        }

        for (int i = 1; i <= _q; i++)
        {
            prediction += _theta[i - 1] * (data[n - i] - prediction);
        }

        return prediction;
    }
}


namespace Example
{
    class Program
    {
        static void Main(string[] args)
            {
                double[] data = new double[400]; // Define the data array.
                double decayRate = 0.01;     // Decay rate of the exponential attenuation

            double sumacc = 2; 
            for (int k = 0; k < 400; k++)
                {
                    Random random = new Random();
                    double randomNumber = random.NextDouble();
                    //var measurement = Math.Sin(k * 3.14 * 5 / 180) * Math.Exp(-decayRate * k) + randomNumber;
                    sumacc = -0.995*sumacc + randomNumber;
                    var measurement = sumacc;
                    if (k > 290 && k < 300) // if k is greater than 50 and less than 60
                        {
                            measurement += -0.5; // add 0.5 to measurement
                        }
                    data[k] = measurement;
                }

                int p = 4; // AR order
                int q = 2; // MA order
                int n = 275; // Use data up to k = 50 to fit the model

                // Extract the data up to k = 50.
                double[] dataToFit = new double[n];
                for (int i = 0; i < n; i++)
                {
                    dataToFit[i] = data[i];
                }

                Vector<double> input = Vector<double>.Build.DenseOfArray(dataToFit);

                ARMA model = new ARMA(p, q);
                double lambda = 0.1;
                model.Fit(input,lambda);

                Console.WriteLine("AR coefficients: " + model._phi.ToString());
                Console.WriteLine("MA coefficients: " + model._theta.ToString());
                Console.ReadLine();

                GraphPane myPane = new GraphPane(new RectangleF(0, 0, 3200, 2400), "Time series", "number", "measurement");
                PointPairList measurementsPairs = new PointPairList();
                PointPairList statesPairs = new PointPairList();
                for (int i = 10; i < 400; i++)
                {
                    measurementsPairs.Add(i, data[i]);
                }
                for (int i = 10; i < 400; i++)
                {
                    double[] selectedData = new double[11];
                    for (int k = i - 10; k <= i; k++)
                    {
                        selectedData[k - i + 10] = -data[k];
                    }
                    Vector<double> input2 = Vector<double>.Build.DenseOfArray(selectedData);
                    double pred_val = model.PredictNext(input2);
                    statesPairs.Add(i, Math.Abs(data[i] - pred_val));
                }
            
                //myPane.AddCurve("measurement", measurementsPairs, Color.Red, SymbolType.Circle);
                myPane.AddCurve("prediction error", statesPairs, Color.Green, SymbolType.XCross);
                Bitmap bm = new Bitmap(200, 200);
                Graphics g = Graphics.FromImage(bm);
                myPane.AxisChange(g);
                Image im = myPane.Image;
                im.Save("result.png", ImageFormat.Png);
        }
    }
}
